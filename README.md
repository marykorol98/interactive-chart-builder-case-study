# Interactive Chart Builder — Case Study
## Overview

This repository presents a **sanitized case study** of an interactive, UI-driven chart builder feature developed for a multi-service platform SMILE.

The feature allows users to **create and configure data visualizations directly within the application** by selecting parameters through a graphical interface.

All materials in this repository are anonymized and intentionally simplified to safely demonstrate the **engineering approach, architecture, and decision-making**, without exposing proprietary or sensitive information.

## Context

The original feature was developed as part of a **large, multi-service production platform** with multiple data sources and user-facing analytical tools.

The goal was to enable non-technical users to **build custom charts on demand** inside the application, without writing code or relying on preconfigured dashboards.

## Problem Statement

Static or pre-built charts were insufficient because:

- users needed flexibility to explore data ad hoc

- different teams required different parameter combinations

- maintaining a growing number of predefined charts did not scale

- incorrect configurations could easily lead to invalid or expensive queries

The challenge was to design a solution that balanced **flexibility, usability, and system safety**.

## Solution Overview

The solution was an **in-application interactive chart builder**, implemented as a product feature rather than a standalone visualization component.

Key ideas:

charts are defined by a **configuration model**, not hardcoded logic

users build charts through a guided UI, with validated parameter choices

the system translates user configuration into a chart specification and data query

invalid or unsafe configurations are prevented early via validation rules

## User Flow
