# SHEild AI

## AI-Powered Women and Children Safety Intelligence Platform

SHEild AI is an intelligent safety platform that leverages crime analytics, predictive intelligence, and smart route planning to help women and children make safer travel decisions. The platform combines official crime statistics, Google Maps services, OpenStreetMap data, and machine learning techniques to analyze city safety, recommend safer travel routes, and provide actionable safety insights through an interactive dashboard.

Unlike conventional navigation applications that primarily optimize routes based on travel time or distance, SHEild AI introduces a **Safety First Navigation System**. Every route is evaluated using historical crime patterns, nearby high-risk locations, road characteristics, and street lighting information before being recommended to the user.

The platform also provides comprehensive crime analytics, city-wise safety scores, AI-generated recommendations, predictive crime trends, and emergency support information, making it a complete women and children safety intelligence platform.

---

# Table of Contents

* [Project Overview](#project-overview)
* [Problem Statement](#problem-statement)
* [Solution](#solution)
* [Key Features](#key-features)
* [System Architecture](#system-architecture)
* [Technology Stack](#technology-stack)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Environment Variables](#environment-variables)
* [Running the Application](#running-the-application)
* [Dashboard Modules](#dashboard-modules)
* [Safe Route Planner](#safe-route-planner)
* [Risk Assessment Engine](#risk-assessment-engine)
* [AI Recommendation System](#ai-recommendation-system)
* [Crime Prediction Module](#crime-prediction-module)
* [APIs Used](#apis-used)
* [Future Improvements](#future-improvements)
* [License](#license)

---

# Project Overview

SHEild AI is a data-driven safety intelligence platform developed to improve personal safety during travel by combining **crime analytics**, **geospatial intelligence**, **machine learning**, and **real-time route analysis**.

The project consists of two major components:

**1. Crime Intelligence Dashboard**

The dashboard analyzes crime statistics across cities and generates meaningful safety insights using historical crime data. It calculates city-wise risk scores, predicts future crime trends, visualizes safety metrics, and provides personalized safety recommendations.

**2. Smart Safe Route Planner**

The route planner recommends safer travel paths by analyzing multiple routes instead of simply selecting the fastest one. Each route is evaluated using crime exposure, nearby high-risk cities, street lighting, road classifications, and travel time before assigning an overall safety score.

The platform is built using **Python**, **Streamlit**, **Google Maps APIs**, **OpenStreetMap**, **Scikit-learn**, **Pandas**, and **Plotly** to deliver an interactive, intelligent, and user-friendly safety application.

---

# Problem Statement

Most navigation applications prioritize **shortest distance** or **fastest travel time** while completely ignoring personal safety. For women and children, especially during night travel or in unfamiliar locations, choosing the fastest route may not always be the safest decision.

Similarly, although government agencies publish annual crime statistics, these datasets are often difficult for the general public to interpret and utilize effectively. Users lack a centralized platform that converts complex crime data into meaningful safety insights and practical travel recommendations.

These limitations create a need for a platform capable of integrating crime intelligence with navigation systems to support informed and safer travel decisions.

---

# Solution

SHEild AI addresses these challenges by integrating **crime analytics**, **geospatial intelligence**, and **predictive analytics** into a unified safety platform.

The platform computes safety scores for cities using historical crime statistics, identifies high-risk regions, predicts future crime trends, and visualizes important safety metrics through an interactive dashboard.

For travel planning, the system generates multiple route alternatives and evaluates each route using several safety parameters including crime exposure, nearby high-risk cities, street lighting conditions, road classifications, and estimated travel duration. Users can adjust the balance between travel speed and safety, allowing the system to recommend routes that best match their preferences.

This approach enables users to make safer travel decisions based on data rather than relying solely on travel time.

---

# Key Features

## Crime Intelligence Dashboard

**City-wise Risk Analysis**

Calculates comprehensive risk scores for each city using multiple crime indicators.

**Safety Score Generation**

Converts calculated risk values into an easy-to-understand safety score ranging from **0 to 100**.

**Risk Classification**

Automatically classifies cities into:

* Low Risk
* Medium Risk
* High Risk

**Interactive Visualizations**

Provides dynamic charts and analytics including:

* Crime trends
* Risk distribution
* Safety comparison
* Top risky cities
* Safest cities
* Dashboard KPIs

**Future Crime Prediction**

Estimates crime statistics for the following year using historical crime trends.

**Personalized Safety Recommendations**

Generates recommendations according to the calculated safety level of each city.

---

## Smart Safe Route Planner

The Safe Route Planner extends traditional navigation by considering safety as the primary optimization objective.

### Route Generation

Multiple route alternatives are generated between the selected origin and destination.

### Geocoding

Converts user-entered locations into geographic coordinates using Google Geocoding API with OpenStreetMap fallback support.

### Route Retrieval

Retrieves driving routes using Google Routes API while supporting OSRM as a fallback routing engine.

### Crime Exposure Analysis

Measures the proximity of each route to nearby high-risk cities and computes a weighted crime exposure score.

### Street Lighting Analysis

Analyzes OpenStreetMap data to estimate the availability of street lighting along the selected route.

### Road Classification

Evaluates different road categories including highways, primary roads, secondary roads, service roads, and isolated roads to improve route safety estimation.

### Safety Ranking

Ranks every available route according to:

* Crime exposure
* Nearby risky cities
* Street lighting
* Road type
* Travel duration

Users can choose between the **Fastest Route** and the **Safest Route** depending on their travel preferences.
