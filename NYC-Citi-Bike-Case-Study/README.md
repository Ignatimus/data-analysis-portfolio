# Cyclistic - Google Business Intelligence Capstone Project

**By Anatoli Ignatov | December 2025**

## 🗂️ About the repo
kur

## 📎 Datasets
Available on **Google Cloud**: 
[NYC Citi Bike Trips](https://console.cloud.google.com/marketplace/details/city-of-new-york/nyc-citi-bike?project=valid-imagery-386312), [Census Bureau US Boundaries](https://console.cloud.google.com/marketplace/product/united-states-census-bureau/us-geographic-boundaries?project=valid-imagery-386312), [GSOD](https://console.cloud.google.com/marketplace/details/noaa-public/gsod?project=valid-imagery-386312) & [Cyclistic NYC zip codes](https://docs.google.com/spreadsheets/d/1IIbH-GM3tdmM5tl56PHhqI7xxCzqaBCU0ylItxk_sy0/template/preview#gid=806359255)

## 🛠️ Tools Used
* BigQuery
* Tableau

## ❓ Business Problem
The company’s Customer Growth Team is creating a business plan for next year. They want to understand how their customers are using their bikes; **Their top priority is identifying customer demand at different station locations.**

## Querying the Data
```sql
SELECT
  TRI.usertype,
  ZIPSTART.zip_code AS zip_code_start,
  ZIPSTARTNAME.borough borough_start,
  ZIPSTARTNAME.neighborhood AS neighborhood_start,
  ZIPEND.zip_code AS zip_code_end,
  ZIPENDNAME.borough borough_end,
  ZIPENDNAME.neighborhood AS neighborhood_end,
  
  -- I add 5 years to make it look recent
  DATE_ADD(DATE(TRI.starttime), INTERVAL 5 YEAR) AS start_day,
  DATE_ADD(DATE(TRI.stoptime), INTERVAL 5 YEAR) AS stop_day,
  WEA.temp AS day_mean_temperature, -- Mean temp
  WEA.wdsp AS day_mean_wind_speed, -- Mean wind speed
  WEA.prcp day_total_precipitation, -- Total precipitation
  
  -- I Group trips into 10 minute intervals to reduces the number of rows
  ROUND(CAST(TRI.tripduration / 60 AS INT64), -1) AS trip_minutes,
  COUNT(TRI.bikeid) AS trip_count
FROM
  `bigquery-public-data.new_york_citibike.citibike_trips` AS TRI
INNER JOIN
  `bigquery-public-data.geo_us_boundaries.zip_codes` ZIPSTART
  ON ST_WITHIN(
    ST_GEOGPOINT(TRI.start_station_longitude, TRI.start_station_latitude),
    ZIPSTART.zip_code_geom)
INNER JOIN
  `bigquery-public-data.geo_us_boundaries.zip_codes` ZIPEND
  ON ST_WITHIN(
    ST_GEOGPOINT(TRI.end_station_longitude, TRI.end_station_latitude),
    ZIPEND.zip_code_geom)
INNER JOIN
  `bigquery-public-data.noaa_gsod.gsod20*` AS WEA
  ON PARSE_DATE("%Y%m%d", CONCAT(WEA.year, WEA.mo, WEA.da)) = DATE(TRI.starttime)
INNER JOIN
  -- Note! Add your zip code table name, enclosed in backticks: `example_table`
  `(insert your table name) zipcodes` AS ZIPSTARTNAME
  ON ZIPSTART.zip_code = CAST(ZIPSTARTNAME.zip AS STRING)
INNER JOIN
  -- Note! Add your zipcode table name, enclosed in backticks: `example_table`
  `(insert your table name) zipcodes` AS ZIPENDNAME
  ON ZIPEND.zip_code = CAST(ZIPENDNAME.zip AS STRING)
WHERE
  -- This takes the weather data from one weather station
  WEA.wban = '94728' -- NEW YORK CENTRAL PARK
  -- Use data from 2014 and 2015
  AND EXTRACT(YEAR FROM DATE(TRI.starttime)) BETWEEN 2014 AND 2015
GROUP BY
  1,
  2,
  3,
  4,
  5,
  6,
  7,
  8,
  9,
  10,
  11,
  12,
  13 
```
## Dashboard - [Link](https://public.tableau.com/views/CyclisticNYCDashboard_17645978518660/Cyclistic-GoogleBusinessIntelligenceCapstoneProject?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

## Dashboard Contents

### Seasonal Trends  
The first tab of the dashboard focuses on seasonality, or trends throughout the year, with the Trip Totals chart and the Trip Counts by Starting Neighborhood table.

[![Cyclistic Dashboard](https://public.tableau.com/static/images/Cy/CyclisticNYCDashboard_17645978518660/Cyclistic-GoogleBusinessIntelligenceCapstoneProject/1.png)](https://public.tableau.com/views/CyclisticNYCDashboard_17645978518660/Cyclistic-GoogleBusinessIntelligenceCapstoneProject)

### Summer Trends
The second tab of the dashboard is a map of seasonal trends of bike trips in each of the New York boroughs. The largest map shows each of the boroughs. The table compares the number of trips and average trip duration for customers and subscribers in each neighborhood. Three smaller maps focus on July, August, and September: the three months with the highest bike traffic.

This map features several filters to focus on specific **user types**, **metrics**, **months**, **starting neighborhoods**, and **ending neighborhoods**. Using any of these filters or clicking on a borough in one of the maps updates the table and maps to focus on your selection in greater detail.

[![Page 2](https://public.tableau.com/static/images/HZ/HZ7HJSP2J/1.png)](https://public.tableau.com/shared/HZ7HJSP2J?:display_count=n&:origin=viz_share_link)

### Top Stations
The third and final tab of the dashboard is a comparison of the total number of trip minutes by starting neighborhood and ending neighborhood for both customers and subscribers. The two charts are horizontal stacked bar graphs that are ordered from highest to lowest number of minutes (between customers and subscribers combined).

kur

### Dashboard Pages

#### Page 1
[![Page 1](https://public.tableau.com/static/images/3Y/3YRXP4N99/1.png)](https://public.tableau.com/shared/3YRXP4N99?:display_count=n&:origin=viz_share_link)

#### Page 2
[![Page 2](https://public.tableau.com/static/images/HZ/HZ7HJSP2J/1.png)](https://public.tableau.com/shared/HZ7HJSP2J?:display_count=n&:origin=viz_share_link)


