# Cyclistic - Google Business Intelligence Capstone Project

**By Anatoli Ignatov | December 2025**

[Tableau Public Dashboard](https://public.tableau.com/views/CyclisticNYCDashboard_17645978518660/Cyclistic-GoogleBusinessIntelligenceCapstoneProject?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

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
-- SELECT CLAUSE: Defining output columns
SELECT 
    -- User information
    TRI.usertype,
    
    -- Starting location details
    ZIPSTART.zip_code AS zip_code_start,
    ZIPSTARTNAME.borough borough_start,
    ZIPSTARTNAME.neighborhood AS neighborhood_start,
    
    -- Ending location details
    ZIPEND.zip_code AS zip_code_end,
    ZIPENDNAME.borough borough_end,
    ZIPENDNAME.neighborhood AS neighborhood_end,
    
    -- Trip timing (Adjusting 5 years forward for dashboarding)
    DATE_ADD(DATE(TRI.starttime), INTERVAL 5 YEAR) AS start_day,
    DATE_ADD(DATE(TRI.stoptime), INTERVAL 5 YEAR) AS stop_day,
    
    -- Weather conditions
    WEA.temp AS day_mean_temperature,
    WEA.wdsp AS day_mean_wind_speed,
    WEA.prcp day_total_precipitation,
    
    -- Trip metrics (Grouping trips into 10 minute intervals to reduces the number of rows)
    ROUND(CAST(TRI.tripduration / 60 AS INT64), -1) AS trip_minutes,
    COUNT(TRI.bikeid) AS trip_count

-- PRIMARY DATA SOURCE: Citibike trip records
FROM 
    bigquery-public-data.new_york_citibike.citibike_trips AS TRI

-- GEOGRAPHIC JOINS: Match stations to zip codes

    -- Join start station coordinates to zip code boundaries
INNER JOIN 
    bigquery-public-data.geo_us_boundaries.zip_codes ZIPSTART 
    ON ST_WITHIN(
        ST_GEOGPOINT(TRI.start_station_longitude, TRI.start_station_latitude),
        ZIPSTART.zip_code_geom)

    -- Join end station coordinates to zip code boundaries
INNER JOIN 
    bigquery-public-data.geo_us_boundaries.zip_codes ZIPEND 
    ON ST_WITHIN(
        ST_GEOGPOINT(TRI.end_station_longitude, TRI.end_station_latitude),
        ZIPEND.zip_code_geom)

-- WEATHER DATA JOIN: Daily conditions from Central Park station
INNER JOIN 
    bigquery-public-data.noaa_gsod.gsod20* AS WEA 
    ON PARSE_DATE("%Y%m%d", CONCAT(WEA.year, WEA.mo, WEA.da)) = DATE(TRI.starttime)

-- NEIGHBORHOOD DETAILS: Add borough and neighborhood names

    -- Add neighborhood details for starting zip code
INNER JOIN 
    `coursera-460808.cyclistic.zip_codes` AS ZIPSTARTNAME 
    ON ZIPSTART.zip_code = CAST(ZIPSTARTNAME.zip AS STRING)

    -- Add neighborhood details for ending zip code
INNER JOIN 
    `coursera-460808.cyclistic.zip_codes` AS ZIPENDNAME 
    ON ZIPEND.zip_code = CAST(ZIPENDNAME.zip AS STRING)

-- FILTERS: Limit data to specific weather station and time period
WHERE 
    -- Use only Central Park weather station data
    WEA.wban = '94728'
    -- Limit to 2014-2015 trip data
    AND EXTRACT(YEAR FROM DATE(TRI.starttime)) BETWEEN 2014 AND 2015

-- GROUPING: Aggregate by all non-aggregated columns
GROUP BY 
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
```
## Final Table

| Variable                | Description                                                |
| ----------------------- | ---------------------------------------------------------- |
| usertype                | Type of user (e.g., subscriber vs customer)                |
| zip_code_start          | ZIP code where the trip started                            |
| borough_start           | Borough where the trip started                             |
| neighborhood_start      | Neighborhood where the trip started                        |
| zip_code_end            | ZIP code where the trip ended                              |
| borough_end             | Borough where the trip ended                               |
| neighborhood_end        | Neighborhood where the trip ended                          |
| start_day               | Date when the trip started                                 |
| stop_day                | Date when the trip ended                                   |
| day_mean_temperature    | Average daily temperature for that date                    |
| day_mean_wind_speed     | Average daily wind speed for that date                     |
| day_total_precipitation | Total daily precipitation for that date                    |
| trip_minutes            | Duration of the trip in minutes                            |
| trip_count              | Number of trips aggregated for that date/route combination |

## Dashboard - [Link](https://public.tableau.com/views/CyclisticNYCDashboard_17645978518660/Cyclistic-GoogleBusinessIntelligenceCapstoneProject?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

### Dashboard Contents

#### Seasonal Trends  
The first tab of the dashboard focuses on seasonality, or trends throughout the year, with the Trip Totals chart and the Trip Counts by Starting Neighborhood table.

**Trips Total Chart**

The Trip Totals chart visualizes the total number of bike trips taken throughout 2019 and 2020, with a distinction between customers and subscribers. This chart shows that subscribers make up a significantly larger portion of Cyclistic’s users than regular customers. It also shows that there are far more users in warmer months (May–October) than there are in colder months. This makes sense considering that people are less likely to ride bicycles in colder weather.
    
This chart was made by putting the Start Day (aggregated by month) in the columns field, the sum of Trip Counts in the rows filed, and UserType as color assignment.
    
**Trip Counts by Starting Neighborhood Table**
    
The Trip Counts by Starting Neighborhood table lists the total number of bike trips started in each neighborhood in each month of 2019 and 2020. It is organized by zip code, borough, and neighborhood. It also uses a color gradient to emphasize the highest and lowest counts of monthly trips. The greater the number of trips, the lighter the value is in the table. It features a colorblind friendly colorscheme so table is readable and accessible.
    
Because the starting location is more indicative of where users look for a bike, it is more important to emphasize starting location when determining where to advertise. The most active stations are in the Lower East Side and the Chelsea and Clinton neighborhoods. The most active months are from May to October.
    
This table was created by putting the Start Day dimension (aggregated by Year and Month) in the Columns field, then the Borough Start and Neighborhood Start dimensions in the Rows field. Then, the color and labels can be set by putting the sum of the Trip Count measure into the Color and Label fields.

#### Summer Trends
The second tab of the dashboard is a map of seasonal trends of bike trips in each of the New York boroughs. The largest map shows each of the boroughs. The table compares the number of trips and average trip duration for customers and subscribers in each neighborhood. Three smaller maps focus on July, August, and September: the three months with the highest bike traffic.

This map features several filters to focus on specific **user types**, **metrics**, **months**, **starting neighborhoods**, and **ending neighborhoods**. Using any of these filters or clicking on a borough in one of the maps updates the table and maps to focus on your selection in greater detail.

#### Top Stations
The third and final tab of the dashboard is a comparison of the total number of trip minutes by starting neighborhood and ending neighborhood for both customers and subscribers. The two charts are horizontal stacked bar graphs that are ordered from highest to lowest number of minutes (between customers and subscribers combined).

These charts lend insight into which locations users are most willing to travel long distances to. The charts show that the Lower East Side and Chelsea and Clinton neighborhoods have the highest total trip minutes for both start and end stations. 

To make the starting neighborhood chart, you can put the sum of Trip Minutes in the columns field, and then the Zip Code Start, Neighborhood Start, and Borough Start dimensions in the rows field. Then, set UserType as the color assignments. To make the ending neighborhood chart, complete the same steps but use the Zip Code End, Neighborhood End, and Borough End dimensions.
