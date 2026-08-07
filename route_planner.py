"""
route_planner.py
==========================================================
Safe Route Planner engine for SHEild AI.

Given an origin and destination, this module:
  1. Geocodes free-text addresses using Google Geocoding API.
  2. Fetches driving route alternatives from Google Routes API.
  3. Scores each alternative by proximity to NCRB-covered cities,
     weighted by each city's computed risk_score.
  4. Analyses street lighting / road type via OpenStreetMap Overpass.
  5. Ranks the alternatives on a Fastest <-> Safest spectrum.
  6. Extracts full turn-by-turn navigation steps for the live
     navigation assistant widget.

API keys are read from environment variables (loaded via .env):
  GOOGLE_MAPS_API_KEY     – used for Maps JS embed in the browser
  GOOGLE_ROUTES_API_KEY   – used for Routes API (backend)
  GOOGLE_GEOCODING_API_KEY – used for Geocoding API (backend)

If a specific key variable is missing the loader falls back to
GOOGLE_MAPS_API_KEY so a single key is sufficient.
==========================================================
"""

import math
import os
import requests
from dotenv import load_dotenv
load_dotenv('env')  # loads 'env' file from the project root

# ----------------------------------------------------------
# API KEY RESOLUTION
# ----------------------------------------------------------

def _key(env_var: str) -> str:
    """Return the value of env_var, falling back through the user's key names."""
    # User's env file uses: maps_api, routes_api, geocoding_api
    fallback_map = {
        "GOOGLE_GEOCODING_API_KEY": "geocoding_api",
        "GOOGLE_ROUTES_API_KEY": "routes_api",
        "GOOGLE_MAPS_API_KEY": "maps_api",
    }
    return (
        os.environ.get(env_var)
        or os.environ.get(fallback_map.get(env_var, ""), "")
        or os.environ.get("maps_api", "")
    )


# ----------------------------------------------------------
# GOOGLE GEOCODING API
# ----------------------------------------------------------

GEOCODING_URL = "https://maps.googleapis.com/maps/api/geocode/json"

# Nominatim fallback (used only when no Google key is configured)
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_NOMINATIM_HEADERS = {"User-Agent": "SHEildAI-SafeRoutePlanner/1.0"}


def _geocode_google(query: str) -> tuple | None:
    """Geocode via Google Geocoding API. Returns (lat, lon) or None."""
    key = _key("GOOGLE_GEOCODING_API_KEY")
    if not key:
        return None
    try:
        resp = requests.get(
            GEOCODING_URL,
            params={"address": query, "key": key, "region": "in"},
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "OK":
            return None
        loc = data["results"][0]["geometry"]["location"]
        return float(loc["lat"]), float(loc["lng"])
    except (requests.RequestException, ValueError, KeyError, IndexError):
        return None


def _geocode_nominatim_attempt(query: str, bias_coords=None) -> tuple | None:
    """OSM Nominatim fallback geocoder (used when no Google key is set)."""
    params = {"q": query, "format": "json", "limit": 1, "countrycodes": "in"}
    if bias_coords:
        lat, lon = bias_coords
        pad = 1.0
        params["viewbox"] = f"{lon - pad},{lat + pad},{lon + pad},{lat - pad}"
        params["bounded"] = 0
    try:
        resp = requests.get(
            NOMINATIM_URL, params=params, headers=_NOMINATIM_HEADERS, timeout=8
        )
        resp.raise_for_status()
        results = resp.json()
        if not results:
            return None
        return float(results[0]["lat"]), float(results[0]["lon"])
    except (requests.RequestException, ValueError, KeyError, IndexError):
        return None


def geocode_place(place_name: str, bias_coords=None) -> tuple | None:
    """
    Convert a free-text place name / address to (lat, lon).

    Tries Google Geocoding API first (if GOOGLE_GEOCODING_API_KEY is set),
    then falls back to OSM Nominatim with progressive query broadening.

    bias_coords: optional (lat, lon) of the other end of the route — used
    by the Nominatim fallback to prefer the correct region for ambiguous
    Indian place names (e.g. "Dwarka" in Delhi vs Gujarat).
    """
    if not place_name or not place_name.strip():
        return None

    # --- Try Google first ---
    result = _geocode_google(place_name)
    if result:
        return result

    # --- OSM Nominatim fallback with progressive broadening ---
    parts = [p.strip() for p in place_name.split(",") if p.strip()]
    if not parts:
        return None
    candidates = [", ".join(parts)]
    for i in range(len(parts) - 1, 0, -1):
        candidates.append(", ".join(parts[:i]))

    for query in candidates:
        result = _geocode_nominatim_attempt(query, bias_coords)
        if result:
            return result
    return None


# ----------------------------------------------------------
# GOOGLE ROUTES API
# ----------------------------------------------------------

ROUTES_API_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"

# OSRM fallback (public demo server, used when no Google key is configured)
OSRM_URL = "http://router.project-osrm.org/route/v1/driving/{coords}"

_MANEUVER_MAP = {
    "TURN_LEFT": "Turn left",
    "TURN_RIGHT": "Turn right",
    "TURN_SHARP_LEFT": "Turn sharp left",
    "TURN_SHARP_RIGHT": "Turn sharp right",
    "TURN_SLIGHT_LEFT": "Keep slight left",
    "TURN_SLIGHT_RIGHT": "Keep slight right",
    "MERGE": "Merge",
    "FORK_LEFT": "Keep left at the fork",
    "FORK_RIGHT": "Keep right at the fork",
    "FERRY": "Take the ferry",
    "ROUNDABOUT_LEFT": "At the roundabout, take the exit on the left",
    "ROUNDABOUT_RIGHT": "At the roundabout, take the exit on the right",
    "UTURN_LEFT": "Make a U-turn",
    "UTURN_RIGHT": "Make a U-turn",
    "STRAIGHT": "Continue straight",
    "RAMP_LEFT": "Take the ramp on the left",
    "RAMP_RIGHT": "Take the ramp on the right",
    "DEPART": "Head",
    "NAME_CHANGE": "Continue on",
    "DESTINATION": "Arrive at destination",
}


def _decode_polyline(encoded: str) -> list:
    """Decode a Google encoded polyline string into [(lat, lon), ...]."""
    coords = []
    index = lat = lng = 0
    while index < len(encoded):
        for is_lng in (False, True):
            shift = result = 0
            while True:
                b = ord(encoded[index]) - 63
                index += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            delta = ~(result >> 1) if result & 1 else result >> 1
            if is_lng:
                lng += delta
            else:
                lat += delta
        coords.append((lat / 1e5, lng / 1e5))
    return coords


def _parse_google_routes(data: dict) -> list:
    """
    Parse a Google Routes API v2 response into the canonical route list
    format used by the rest of this module:
      {
        "coords": [(lat, lon), ...],
        "distance_km": float,
        "duration_min": float,
        "road_summary": str,
        "turn_steps": [...],   ← NEW: full navigation step list
      }
    """
    routes = []
    for route in data.get("routes", []):
        polyline_str = route.get("polyline", {}).get("encodedPolyline", "")
        coords = _decode_polyline(polyline_str) if polyline_str else []

        distance_m = route.get("distanceMeters", 0)
        duration_str = route.get("duration", "0s")
        duration_s = int(duration_str.rstrip("s")) if duration_str.endswith("s") else 0

        # Road summary from route description or legs
        description = route.get("description", "")
        if not description:
            labels = []
            for leg in route.get("legs", []):
                for step in leg.get("steps", []):
                    nav = step.get("navigationInstruction", {})
                    road = nav.get("instructions", "")
                    if road and road not in labels:
                        labels.append(road)
                        if len(labels) >= 3:
                            break
            description = ", ".join(labels[:3]) if labels else "Route"

        turn_steps = extract_turn_steps_google(route)

        routes.append({
            "coords": coords,
            "distance_km": round(distance_m / 1000, 2),
            "duration_min": round(duration_s / 60, 1),
            "road_summary": description,
            "turn_steps": turn_steps,
        })
    return routes


def extract_turn_steps_google(route: dict) -> list:
    """
    Extract a clean turn-by-turn step list from a single Google Routes route.

    Each step dict:
      {
        "instruction": str,       # e.g. "Turn left onto Ring Road"
        "distance_m": float,      # metres to travel on this step
        "duration_s": float,      # seconds for this step
        "lat": float,             # start lat of this step
        "lon": float,             # start lon of this step
        "maneuver": str,          # raw maneuver type string
      }
    """
    steps = []
    for leg in route.get("legs", []):
        for step in leg.get("steps", []):
            nav = step.get("navigationInstruction", {})
            maneuver_raw = nav.get("maneuver", "STRAIGHT")
            maneuver_verb = _MANEUVER_MAP.get(maneuver_raw, "Continue")
            road = step.get("localizedValues", {}).get("staticDuration", {})
            # Road name from instructions text
            instruction_text = nav.get("instructions", "")
            if not instruction_text:
                instruction_text = maneuver_verb

            dist_m = step.get("distanceMeters", 0)
            dur_str = step.get("staticDuration", "0s")
            dur_s = int(dur_str.rstrip("s")) if dur_str.endswith("s") else 0

            # Step start coordinates
            start_loc = step.get("startLocation", {}).get("latLng", {})
            s_lat = start_loc.get("latitude", 0.0)
            s_lon = start_loc.get("longitude", 0.0)

            steps.append({
                "instruction": instruction_text,
                "distance_m": dist_m,
                "duration_s": dur_s,
                "lat": s_lat,
                "lon": s_lon,
                "maneuver": maneuver_raw,
            })
    # Append arrival step
    if steps:
        last = steps[-1]
        end_loc = {}
        for leg in route.get("legs", []):
            end_loc = leg.get("endLocation", {}).get("latLng", {})
        steps.append({
            "instruction": "Arrive at your destination",
            "distance_m": 0,
            "duration_s": 0,
            "lat": end_loc.get("latitude", last["lat"]),
            "lon": end_loc.get("longitude", last["lon"]),
            "maneuver": "DESTINATION",
        })
    return steps


def _get_routes_google(origin: tuple, destination: tuple, max_alternatives: int = 3) -> list:
    """Fetch driving routes from the Google Routes API v2."""
    key = _key("GOOGLE_ROUTES_API_KEY")
    if not key:
        return []

    lat1, lon1 = origin
    lat2, lon2 = destination

    body = {
        "origin": {"location": {"latLng": {"latitude": lat1, "longitude": lon1}}},
        "destination": {"location": {"latLng": {"latitude": lat2, "longitude": lon2}}},
        "travelMode": "DRIVE",
        "routingPreference": "TRAFFIC_AWARE",
        "computeAlternativeRoutes": True,
        "routeModifiers": {"avoidTolls": False},
        "languageCode": "en-IN",
        "units": "METRIC",
    }

    try:
        resp = requests.post(
            ROUTES_API_URL,
            json=body,
            headers={
                "X-Goog-Api-Key": key,
                "X-Goog-FieldMask": (
                    "routes.duration,routes.distanceMeters,routes.polyline,"
                    "routes.description,routes.legs.steps.navigationInstruction,"
                    "routes.legs.steps.distanceMeters,routes.legs.steps.staticDuration,"
                    "routes.legs.steps.startLocation,routes.legs.steps.endLocation,"
                    "routes.legs.endLocation"
                ),
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError):
        return []

    return _parse_google_routes(data)[:max_alternatives]


# ----------------------------------------------------------
# OSRM FALLBACK ROUTING (used when no Google key is set)
# ----------------------------------------------------------

def _extract_road_summary(route: dict, max_roads: int = 4) -> str:
    """Build a human-readable 'via X, Y, Z' string from OSRM leg/step names."""
    leg_summaries = [
        leg["summary"].strip()
        for leg in route.get("legs", [])
        if leg.get("summary", "").strip()
    ]
    if leg_summaries:
        return " → ".join(leg_summaries)

    distance_by_name: dict = {}
    for leg in route.get("legs", []):
        for step in leg.get("steps", []):
            name = step.get("name", "").strip()
            if not name:
                continue
            distance_by_name[name] = distance_by_name.get(name, 0) + step.get("distance", 0)

    if not distance_by_name:
        return "Unnamed route"

    top_roads = sorted(distance_by_name.items(), key=lambda x: x[1], reverse=True)[:max_roads]
    return "via " + ", ".join(name for name, _ in top_roads)


_OSRM_MANEUVER_MAP = {
    "turn": {"left": "Turn left", "right": "Turn right",
             "sharp left": "Turn sharp left", "sharp right": "Turn sharp right",
             "slight left": "Keep slight left", "slight right": "Keep slight right"},
    "depart": {"": "Head"},
    "arrive": {"": "Arrive at destination"},
    "merge": {"": "Merge"},
    "roundabout": {"": "Take the roundabout"},
    "rotary": {"": "Take the roundabout"},
    "fork": {"left": "Keep left at the fork", "right": "Keep right at the fork"},
    "new name": {"": "Continue on"},
    "continue": {"": "Continue straight"},
    "end of road": {"left": "Turn left", "right": "Turn right"},
    "ferry": {"": "Take the ferry"},
    "use lane": {"": "Use the lane"},
}


def _extract_turn_steps_osrm(route: dict) -> list:
    """Extract turn steps from an OSRM raw route dict."""
    steps = []
    for leg in route.get("legs", []):
        for step in leg.get("steps", []):
            maneuver = step.get("maneuver", {})
            m_type = maneuver.get("type", "")
            m_mod = maneuver.get("modifier", "")
            road_name = step.get("name", "").strip()

            verb_map = _OSRM_MANEUVER_MAP.get(m_type, {})
            verb = verb_map.get(m_mod) or verb_map.get("") or "Continue"
            instruction = f"{verb} onto {road_name}" if road_name and m_type not in ("arrive", "depart") else verb
            if m_type == "depart" and road_name:
                instruction = f"Head on {road_name}"

            loc = maneuver.get("location", [0, 0])  # [lon, lat]
            steps.append({
                "instruction": instruction,
                "distance_m": step.get("distance", 0),
                "duration_s": step.get("duration", 0),
                "lat": loc[1],
                "lon": loc[0],
                "maneuver": m_type.upper().replace(" ", "_"),
            })
    return steps


def _get_routes_osrm(origin: tuple, destination: tuple, max_alternatives: int = 3) -> list:
    """Fetch routes from the public OSRM demo server (open fallback)."""
    lat1, lon1 = origin
    lat2, lon2 = destination
    coord_str = f"{lon1},{lat1};{lon2},{lat2}"
    url = OSRM_URL.format(coords=coord_str)

    try:
        resp = requests.get(
            url,
            params={
                "alternatives": "true",
                "overview": "full",
                "geometries": "geojson",
                "steps": "true",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError):
        return []

    if data.get("code") != "Ok":
        return []

    routes = []
    for route in data.get("routes", [])[:max_alternatives]:
        line = route["geometry"]["coordinates"]  # [[lon, lat], ...]
        coords = [(lat, lon) for lon, lat in line]
        routes.append({
            "coords": coords,
            "distance_km": round(route["distance"] / 1000, 2),
            "duration_min": round(route["duration"] / 60, 1),
            "road_summary": _extract_road_summary(route),
            "turn_steps": _extract_turn_steps_osrm(route),
        })
    return routes


# ----------------------------------------------------------
# PUBLIC ROUTING ENTRY POINT
# ----------------------------------------------------------

def get_route_alternatives(origin: tuple, destination: tuple, max_alternatives: int = 3) -> list:
    """
    Fetch real road-network driving route alternatives between origin and
    destination (lat, lon tuples).

    Uses Google Routes API when GOOGLE_ROUTES_API_KEY (or GOOGLE_MAPS_API_KEY)
    is set; otherwise falls back to the public OSRM demo server.

    Returns a list of route dicts:
      {
        "coords": [(lat, lon), ...],
        "distance_km": float,
        "duration_min": float,
        "road_summary": str,
        "turn_steps": [{"instruction", "distance_m", "duration_s",
                         "lat", "lon", "maneuver"}, ...],
      }
    Returns [] if both providers fail.
    """
    routes = _get_routes_google(origin, destination, max_alternatives)
    if not routes:
        routes = _get_routes_osrm(origin, destination, max_alternatives)
    return routes


# ==========================================================
# RISK SCORING
# ==========================================================

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points, in km."""
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def score_route_risk(route_coords: list, ncrb_df, influence_radius_km: float = 40, sample_every: int = 3) -> dict:
    """
    Estimates a route's exposure to higher-crime NCRB cities.

    For a sample of points along the route, any covered NCRB city within
    `influence_radius_km` contributes to the score, weighted by proximity
    and that city's own risk_score.

    Returns:
      {
        "avg_risk": float (0-1),
        "nearby_cities": [{"city", "distance_km", "risk_level", "risk_score"}, ...],
        "max_city_risk": float,
      }
    """
    if not route_coords or ncrb_df.empty:
        return {"avg_risk": 0.0, "nearby_cities": [], "max_city_risk": 0.0}

    sampled = route_coords[::sample_every] or route_coords
    nearby: dict = {}
    point_risks: list = []

    for lat, lon in sampled:
        point_weighted_sum = 0.0
        point_weight_total = 0.0

        for _, city_row in ncrb_df.iterrows():
            c_lat, c_lon = city_row.get("lat"), city_row.get("lon")
            if c_lat is None or c_lon is None:
                continue
            if isinstance(c_lat, float) and math.isnan(c_lat):
                continue

            dist = _haversine_km(lat, lon, c_lat, c_lon)
            if dist <= influence_radius_km:
                weight = max(0.0, 1 - (dist / influence_radius_km))
                point_weighted_sum += weight * city_row["risk_score"]
                point_weight_total += weight

                city_name = city_row["city"]
                if city_name not in nearby or dist < nearby[city_name]["distance_km"]:
                    nearby[city_name] = {
                        "distance_km": round(dist, 1),
                        "risk_level": city_row["risk_level"],
                        "risk_score": round(float(city_row["risk_score"]), 2),
                    }

        point_risk = (point_weighted_sum / point_weight_total) if point_weight_total > 0 else 0.0
        point_risks.append(point_risk)

    avg_risk = sum(point_risks) / len(point_risks) if point_risks else 0.0
    max_city_risk = max((v["risk_score"] for v in nearby.values()), default=0.0)

    nearby_list = sorted(
        [{"city": k, **v} for k, v in nearby.items()],
        key=lambda x: x["distance_km"],
    )

    return {
        "avg_risk": round(avg_risk, 6),
        "nearby_cities": nearby_list,
        "max_city_risk": round(max_city_risk, 3),
    }


# ==========================================================
# STREET-LEVEL LIGHTING / ROAD-TYPE SCORING (OpenStreetMap)
# ==========================================================

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]

MAJOR_ROAD_TYPES = {
    "motorway", "trunk", "primary", "secondary",
    "motorway_link", "trunk_link", "primary_link", "secondary_link",
}
QUIET_ROAD_TYPES = {"track", "path", "service", "living_street", "unclassified"}

_MAX_ROUTE_KM_FOR_STREET_CHECK = 60
_MAX_WAYS_CONSIDERED = 2000
_MAX_SAMPLE_POINTS = 25
_MAX_QUERY_POINTS = 80
_AROUND_RADIUS_M = 60


def fetch_street_features(route_coords: list) -> list:
    """
    Query OpenStreetMap Overpass for road segments near the route polyline
    using a corridor query. Returns [] if the route is too long or every
    Overpass mirror fails.
    """
    if not route_coords:
        return []

    total_km = sum(
        _haversine_km(route_coords[i][0], route_coords[i][1],
                      route_coords[i + 1][0], route_coords[i + 1][1])
        for i in range(len(route_coords) - 1)
    )
    if total_km > _MAX_ROUTE_KM_FOR_STREET_CHECK:
        return []

    step = max(1, len(route_coords) // _MAX_QUERY_POINTS)
    query_points = route_coords[::step]
    coord_list = ",".join(f"{lat},{lon}" for lat, lon in query_points)

    query = f"""
    [out:json][timeout:20];
    way["highway"](around:{_AROUND_RADIUS_M},{coord_list});
    out geom;
    """

    data = None
    for url in OVERPASS_URLS:
        try:
            resp = requests.post(url, data={"data": query}, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            break
        except (requests.RequestException, ValueError):
            continue

    if data is None:
        return []

    ways = []
    for el in data.get("elements", [])[:_MAX_WAYS_CONSIDERED]:
        if el.get("type") != "way":
            continue
        geometry = el.get("geometry", [])
        if not geometry:
            continue
        tags = el.get("tags", {})
        pts = [(pt["lat"], pt["lon"]) for pt in geometry]
        way_lats = [p[0] for p in pts]
        way_lons = [p[1] for p in pts]
        ways.append({
            "highway": tags.get("highway", ""),
            "lit": tags.get("lit", ""),
            "points": pts,
            "bbox": (min(way_lats), min(way_lons), max(way_lats), max(way_lons)),
        })
    return ways


def score_route_lighting(route_coords: list, match_radius_km: float = 0.05) -> dict:
    """
    Estimate how well-lit / busy a route is from OSM road segments.

    Returns:
      {
        "dim_score": float 0-1 or None,
        "lit_coverage_pct": float or None,
        "major_road_pct": float or None,
        "available": bool,
      }
    """
    ways = fetch_street_features(route_coords)
    if not ways:
        return {"dim_score": None, "lit_coverage_pct": None,
                "major_road_pct": None, "available": False}

    step = max(1, len(route_coords) // _MAX_SAMPLE_POINTS)
    sampled = route_coords[::step] or route_coords

    lit_yes = lit_no = major = matched = 0
    pad = match_radius_km / 111.0

    for lat, lon in sampled:
        best_dist = None
        best_way = None
        for way in ways:
            wsouth, wwest, wnorth, weast = way["bbox"]
            if not (wsouth - pad <= lat <= wnorth + pad and wwest - pad <= lon <= weast + pad):
                continue
            for wlat, wlon in way["points"]:
                d = _haversine_km(lat, lon, wlat, wlon)
                if best_dist is None or d < best_dist:
                    best_dist = d
                    best_way = way
        if best_way is not None and best_dist is not None and best_dist <= match_radius_km:
            matched += 1
            if best_way["lit"] == "yes":
                lit_yes += 1
            elif best_way["lit"] == "no":
                lit_no += 1
            if best_way["highway"] in MAJOR_ROAD_TYPES:
                major += 1

    if matched == 0:
        return {"dim_score": None, "lit_coverage_pct": None,
                "major_road_pct": None, "available": False}

    lit_known = lit_yes + lit_no
    lit_coverage_pct = round((lit_yes / lit_known) * 100, 1) if lit_known > 0 else None
    major_road_pct = round((major / matched) * 100, 1)
    unlit_component = (lit_no / lit_known) if lit_known > 0 else 0.4
    minor_component = 1 - (major / matched)
    dim_score = round(0.5 * unlit_component + 0.5 * minor_component, 3)

    return {
        "dim_score": dim_score,
        "lit_coverage_pct": lit_coverage_pct,
        "major_road_pct": major_road_pct,
        "available": True,
    }


# ==========================================================
# RANKING
# ==========================================================

def rank_routes(routes: list, ncrb_df, safety_weight: float = 0.5,
                include_street_lighting: bool = True) -> list:
    """
    Score and rank route alternatives.

    safety_weight: 0.0 -> purely fastest/shortest,
                   1.0 -> purely safest,
                   0.5 -> balanced.

    "Safest" blends crime proximity (NCRB) and street lighting/road type
    (OSM) when both are available; falls back to crime data alone otherwise.

    Each route dict is augmented in-place with:
      risk_info, lighting_info, norm_distance, norm_safety,
      combined_score, tags, risk_tied_with_alternative

    Returns the list sorted best-first (lowest combined_score first).
    """
    if not routes:
        return []

    for r in routes:
        r["risk_info"] = score_route_risk(r["coords"], ncrb_df)
        if include_street_lighting:
            r["lighting_info"] = score_route_lighting(r["coords"])
        else:
            r["lighting_info"] = {"dim_score": None, "lit_coverage_pct": None,
                                   "major_road_pct": None, "available": False}

    max_dist = max(r["distance_km"] for r in routes) or 1
    max_crime = max(r["risk_info"]["avg_risk"] for r in routes)
    dim_scores = [r["lighting_info"]["dim_score"] for r in routes
                  if r["lighting_info"]["dim_score"] is not None]
    max_dim = max(dim_scores) if dim_scores else None

    for r in routes:
        norm_dist = r["distance_km"] / max_dist
        norm_crime = (r["risk_info"]["avg_risk"] / max_crime) if max_crime > 0 else 0.0

        dim_score = r["lighting_info"]["dim_score"]
        if dim_score is not None and max_dim and max_dim > 0:
            norm_dim = dim_score / max_dim
            norm_safety = 0.5 * norm_crime + 0.5 * norm_dim
        else:
            norm_safety = norm_crime

        r["norm_distance"] = round(norm_dist, 3)
        r["norm_safety"] = round(norm_safety, 3)
        r["combined_score"] = round(
            (1 - safety_weight) * norm_dist + safety_weight * norm_safety, 4
        )

    ranked = sorted(routes, key=lambda r: r["combined_score"])

    if len(ranked) > 1:
        fastest = min(routes, key=lambda r: r["distance_km"])
        safest = min(routes, key=lambda r: (r["norm_safety"], r["distance_km"]))
        for r in ranked:
            tags = []
            if r is fastest:
                tags.append("Fastest")
            if r is safest:
                tags.append("Safest")
            r["tags"] = tags

        top_safety = ranked[0]["norm_safety"]
        runner_up_safety = ranked[1]["norm_safety"]
        safety_scale = max(top_safety, runner_up_safety, 1e-9)
        is_tied = abs(top_safety - runner_up_safety) / safety_scale < 0.01
        for r in ranked:
            r["risk_tied_with_alternative"] = is_tied
    else:
        ranked[0]["tags"] = ["Only route found"]
        ranked[0]["risk_tied_with_alternative"] = False

    return ranked
