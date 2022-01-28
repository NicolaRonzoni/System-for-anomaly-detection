# Pandas-InfluxDB
Workflow  for time series analysis. 

Data Ingestion on Influxdb from Pandas :
1. Generate API token from UserInterface and create a bucket (copy the ID) where store the time series. 
2. Load a csv file in Pandas. The Pandas dataframe should have as index the timestamp, it must contain a column that indicates the origin* of the data (by default creates a column that contains the same value in each row).
3. Despite InfluxDB is robust to events coming at different intervals, make sense to make the data regular (model development). For this reason using Pandas add rows of missing values if jump in the series occur. 
4. Make a connection with the server
5. Send data, specifying the name of the dataframe and the origin*. (check here available plan https://www.influxdata.com/influxdb-cloud-pricing/ no storage for the free-plan)
6. Close the connection

It is always possible to restore data in Pandas once manipulation are computed on InfluxDB using the same scheme showed above.  


