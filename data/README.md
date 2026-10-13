# Data

The raw data is **not stored in this repository** (hundreds of MB). Download it from the
U.S. Bureau of Transportation Statistics (BTS):

**https://www.transtats.bts.gov/**

## How to download
1. Open the link above → **Data Finder / Aviation → Airline On-Time Statistics and Delay Causes**
   → **Reporting Carrier On-Time Performance (1987–present)** → **Download**.
2. Select **2025** as the year, then download these months: **January, June, November, and December**.
3. Tick the fields listed below, click **Download**, and unzip the CSV.
4. Rename and place the files in `data/raw/` exactly like this:

| Month | File name | Rows used in the study |
|---|---|---:|
| January 2025 | `T_ONTIME_REPORTING1.csv` | 539,747 |
| June 2025 | `T_ONTIME_REPORTING6.csv` | 611,575 |
| November 2025 | `T_ONTIME_REPORTING11.csv` | 570,550 |
| December 2025 | `T_ONTIME_REPORTING12.csv` | 582,304 |

Jan + Jun + Nov = 1,721,872 rows (used for the 300K stratified sample, train/val/Nov-test).
December (582,304 rows) is a separate, untouched holdout.

> Row counts depend on the exact BTS extract and download date. Small differences are normal.

## Fields to select
`YEAR, MONTH, FL_DATE, OP_UNIQUE_CARRIER, TAIL_NUM, OP_CARRIER_FL_NUM, ORIGIN_AIRPORT_ID,
ORIGIN, ORIGIN_CITY_NAME, ORIGIN_STATE_ABR, DEST_AIRPORT_ID, DEST_AIRPORT_SEQ_ID, DEST,
DEST_CITY_NAME, DEST_STATE_ABR, CRS_DEP_TIME, DEP_TIME, DEP_DELAY, DEP_DELAY_GROUP,
CRS_ARR_TIME, ARR_TIME, ARR_DELAY, ARR_DELAY_GROUP, CANCELLED, CANCELLATION_CODE, DIVERTED,
CRS_ELAPSED_TIME, AIR_TIME, FLIGHTS, DISTANCE, CARRIER_DELAY, WEATHER_DELAY, NAS_DELAY,
SECURITY_DELAY, LATE_AIRCRAFT_DELAY`

(The notebooks need at least: `MONTH, FL_DATE, OP_UNIQUE_CARRIER, TAIL_NUM, OP_CARRIER_FL_NUM,
ORIGIN, ORIGIN_STATE_ABR, DEST, DEST_STATE_ABR, CRS_DEP_TIME, DEP_TIME, DEP_DELAY,
DEP_DELAY_GROUP, CRS_ARR_TIME, ARR_TIME, ARR_DELAY, ARR_DELAY_GROUP, CANCELLED,
CANCELLATION_CODE, DIVERTED, CRS_ELAPSED_TIME, AIR_TIME, DISTANCE, YEAR` and the five delay-cause columns.)
