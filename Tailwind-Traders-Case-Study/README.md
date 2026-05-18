# Tailwind Traders Power BI Dashboard

This project is my end-to-end Power BI case study built around the Tailwind Traders sales dataset. I prepared the source data in Excel, loaded and cleaned multiple tables in Power BI, built a relational model, created DAX measures, and designed sales and profit reports.

## Project Highlights

- Prepared and cleaned sales data in Excel
- Loaded and optimized multiple data sources in Power BI
- Built a star-like model with supporting lookup tables
- Created a Calendar table for time intelligence
- Converted sales values into USD for consistent analysis
- Built DAX measures for profit analysis and median sales
- Designed two report pages:
  - Sales Overview
  - Profit Overview
- Published the final report to Power BI Service

## Data Sources

I worked with these tables and inputs:

- Sales
- Purchases
- Countries
- Historical Currency Exchange data
- CalendarTable
- Sales in USD

## Data Preparation

I transformed the Sales table in Excel by adding calculated columns for:

- Cost per Unit
- Gross Revenue
- Total Tax
- Net Revenue
- Profit

In Power Query, I standardised column data types, checked data quality, and reviewed column distributions and profiles to validate the data before modeling.

I also loaded the historical exchange-rate data using a Python script so I could convert all sales values into USD.

## Data Model

I built the following relationships in the model:

- Countries ↔ Exchange Data on Exchange ID
- Sales ↔ Countries on Country ID
- Purchases ↔ Sales on OrderID
- CalendarTable ↔ Purchases on Date
- Sales in USD ↔ Sales on OrderID

I created a Sales in USD calculated table to make cross-currency analysis easier and more consistent across the report.

## DAX Measures

I created the following measures for time intelligence and reporting:

```DAX
Yearly Profit Margin =
DIVIDE(
    SUM('Sales in USD'[Profit USD]),
    SUM('Sales in USD'[Net Revenue USD])
)

Quarterly Profit Margin =
CALCULATE(
    [Yearly Profit Margin],
    DATESQTD('CalendarTable'[Date])
)

YTD Profit Margin =
TOTALYTD([Yearly Profit Margin], 'CalendarTable'[Date])

Median Sales =
MEDIAN('Sales in USD'[Gross Revenue USD])
