from collections import Counter
import json
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_replace
import matplotlib.pyplot as plt
from pyspark.sql.functions import count
from pyspark.sql.functions import col
from pyspark.sql.functions import year, month, to_timestamp, sum, dayofmonth, avg, hour, dayofweek
from pyspark.sql.functions import when
from folium.plugins import MarkerCluster
from matplotlib.colors import to_hex
import folium
import seaborn as sns
import pandas as pd
import numpy as np
from pyspark.ml.classification import LogisticRegression
import streamlit as st
import geopandas as gpd
from shapely import wkt
from streamlit_option_menu import option_menu
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, MinMaxScaler, OneHotEncoder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.clustering import KMeans
from pyspark.ml.stat import Correlation

from pyspark.ml.evaluation  import RegressionEvaluator

#from pyspark.sql.functions import udf

from pyspark.sql.types import  FloatType, StructField, StructType
from pyspark.sql.window import Window


def convert_to_geojson():
    df = pd.read_csv("dataset_files/PoliceDistrictDec2012_20250127.csv")
    # Converto la colonna 'geometry' da WKT a oggetti shapely
    df['geometry'] = df['the_geom'].apply(wkt.loads)
    # Converto il DataFrame in un GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    # Salvo il GeoDataFrame in formato GeoJSON
    gdf.to_file("output.geojson", driver="GeoJSON")

          
def file_exists(self, path):
    """
    Verifica se un file o una cartella esiste su HDFS utilizzando l'API Java di Hadoop.
    """
    hadoop_conf = self.spark._jsc.hadoopConfiguration()
    fs = self.spark._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)
    return fs.exists(self.spark._jvm.org.apache.hadoop.fs.Path(path))


class Dataset:

    def __init__(self):
        self.spark = SparkSession.builder.master("local[*]").appName("Progetto").config("spark.executor.memory", "10g").config("spark.driver.memory", "6g").config("spark.driver.maxResultSize", "2g")\
            .config("spark.executor.cores", "4").config("spark.sql.shuffle.partitions", "200").config("spark.default.parallelism", "200").config("spark.kryoserializer.buffer.max", "1024m").config("spark.hadoop.fs.hdfs.impl", "org.apache.hadoop.hdfs.DistributedFileSystem") \
            .config("spark.hadoop.io.native.lib", "false") \
            .getOrCreate()
        #if "df_cleaned" not in st.session_state:
        df_cleaned_path = "dataset_files\df_cleaned.parquet"
        df_season_path = "dataset_files\df_season.parquet"
        df_arrested_path = "dataset_files\df_arrested.parquet"
        if file_exists(self, df_cleaned_path) and file_exists(self, df_season_path) and file_exists(self, df_arrested_path):
            self.df_cleaned = self.spark.read.parquet(df_cleaned_path)
            self.df_season = self.spark.read.parquet(df_season_path)
            self.df_arrested = self.spark.read.parquet(df_arrested_path)
            self.null_counts = self.spark.read.parquet("dataset_files/null_counts.parquet")
        else:
            # Leggo il file CSV
            self.df = self.spark.read.csv("dataset_files\Crimes_-_2001_to_Present.csv", header=True, inferSchema=True)
            #conto i valori nulli presenti nel dataset
            self.null_counts = self.df.select([sum(col(c).isNull().cast("int")).alias(c) for c in self.df.columns])
            self.null_counts.write.mode("overwrite").parquet("dataset_files/null_counts.parquet")
            
            #elimino i valori null
            self.df_cleaned = self.df.dropna()

            self.df_cleaned = self.df_cleaned.drop_duplicates()
            #elimino -/: 
            self.df_cleaned = self.df_cleaned.withColumn("Description", regexp_replace("Description", "[-/:]", " "))
            #elimino location perchè ridondante
            self.df_cleaned = self.df_cleaned.drop('Location')
            self.df_cleaned = self.df_cleaned.withColumn(
                "Date",
                to_timestamp(col("Date"), "MM/dd/yyyy hh:mm:ss a")
            )

            self.df_cleaned = self.df_cleaned.withColumn("month", month(col("Date")))
            self.df_cleaned = self.df_cleaned.withColumn("year", year(col("Date")))
            self.df_cleaned = self.df_cleaned.withColumn("day", dayofmonth(col("Date")))
            self.df_cleaned = self.df_cleaned.withColumn("hour", hour(col("Date")))
            self.df_cleaned = self.df_cleaned.withColumn("dayofweek", dayofweek(col("Date")))
            self.df_cleaned = self.df_cleaned.drop("Date")
            self.df_cleaned = self.df_cleaned.filter(self.df_cleaned['year']%4==0)
            # pochi dati nel 2001/2002 e nel 2021
            self.df_cleaned = self.df_cleaned.filter((col('year') > 2002) & (col('year') < 2021))

            self.df_cleaned = self.df_cleaned.drop("Block")
            #self.df_cleaned = self.df_cleaned.drop("FBI Code")
            self.df_cleaned = self.df_cleaned.drop("IUCR")
            self.df_cleaned = self.df_cleaned.drop("Beat")
            self.df_cleaned = self.df_cleaned.drop("Ward")
            self.df_cleaned = self.df_cleaned.drop("X Coordinate")
            self.df_cleaned = self.df_cleaned.drop("Y Coordinate")

            self.df_cleaned= self.df_cleaned.withColumn("Arrest", when(col("Arrest") == True, 1).otherwise(0))
            self.df_cleaned= self.df_cleaned.withColumn("Domestic", when(col("Arrest") == True, 1).otherwise(0))
            #self.df_cleaned.printSchema()

            # Creo una vista temporanea
            #self.df_cleaned.createOrReplaceTempView("table")
            #df_cleaned.show()

            self.df_arrested = self.df_cleaned.filter(col("Arrest") == True)

            self.df_season = self.df_cleaned.withColumn(
                "season",
                when((col("month") == 12) | (col("month") == 1) | (col("month") == 2), "Winter")
                .when((col("month") == 3) | (col("month") == 4) | (col("month") == 5), "Spring")
                .when((col("month") == 6) | (col("month") == 7) | (col("month") == 8), "Summer")
                .otherwise("Autumn")
            )
            crime_types_to_remove = [
                "CONCEALED CARRY LICENSE VIOLATION",
                "OBSCENITY",
                "PUBLIC INDECENCY",
                "NON-CRIMINAL",
                "OTHER NARCOTIC VIOLATION",
                "HUMAN TRAFFICKING",
                "NON - CRIMINAL",
                "RITUALISM",
                "NON-CRIMINAL (SUBJECT SPECIFIED)"
            ]

            # Filtra il DataFrame per rimuovere le righe indesiderate
            self.df_cleaned = self.df_cleaned.filter(
                ~col("Primary Type").isin(crime_types_to_remove)
            )

            
            # Cache del DataFrame pulito
            #self.df_cleaned.cache()
            #self.df_season.cache()
            #self.df_arrested.cache()

            # Salvo il DataFrame pulito in session_state
            #st.session_state["df_cleaned"] = self.df_cleaned
            #st.session_state["df_season"] = self.df_season
            #st.session_state["df_arrested"] = self.df_arrested
    #else:
            # Recupero il DataFrame dalla session_state
            #self.df_cleaned = st.session_state["df_cleaned"]
            #self.df_season = st.session_state.get("df_season", self.df_cleaned.withColumn(
            #    "season",
            #    when((col("month") == 12) | (col("month") == 1) | (col("month") == 2), "Winter")
            #    .when((col("month") == 3) | (col("month") == 4) | (col("month") == 5), "Spring")
            #    .when((col("month") == 6) | (col("month") == 7) | (col("month") == 8), "Summer")
            #    .otherwise("Autumn")
            #))
            self.df_arrested = self.df_cleaned.filter(col("Arrest") == True)
            self.df_season.write.mode("overwrite").parquet("dataset_files\df_season.parquet")
            self.df_arrested.write.mode("overwrite").parquet("dataset_files\df_arrested.parquet")
            self.df_cleaned.write.mode("overwrite").parquet("dataset_files\df_cleaned.parquet")


    def showCriticalHour(self):
        # Specifica il percorso HDFS dove salvare il CSV del risultato aggregato
        parquet_path = "dataset_files\max_crime_for_hour.parquet"
        if file_exists(self,parquet_path):
            # Se il file esiste, lo leggo direttamente
            max_crime_for_hour = self.spark.read.parquet(parquet_path)
        else:
            crime_counts = self.df_arrested.groupBy("hour", "Primary Type").count()
            max_crime_for_hour = crime_counts.groupBy("hour").agg({"count": "max"})
            max_crime_for_hour.write.mode("overwrite").parquet(parquet_path)

        max_crime_for_hour = max_crime_for_hour.toPandas()
        st.subheader("🕐Trend of crimes over the day")
        
        # Creiamo il grafico a linee
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(x=max_crime_for_hour["hour"], y=max_crime_for_hour["max(count)"], marker="o", color="blue", linewidth=2.5, ax=ax)

        # Formattiamo il grafico
        ax.set_xticks(range(0, 24))  # Mostriamo tutte le ore da 0 a 23
        ax.set_xlabel("Dayhour")
        ax.set_ylabel("Number of Crimes")
        ax.set_title("Trend of crimes over the day")
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        st.pyplot(fig)


    def showPlotPrimaryType(self):
        parquet_pathplot = "dataset_files\plotPrimaryType.parquet"
        if file_exists(self,parquet_pathplot):
            # Se il file (o meglio, la directory CSV) esiste, lo leggo direttamente 
            crime_counts = self.spark.read.parquet(parquet_pathplot)
        else:
            #prendo la colonna primary key
            self.df_cleaned = self.df_cleaned.withColumn('Primary Type', col('Primary Type'))
            # ottengo i valori unici della colonna 'Primary key'
            unique_values = self.df_cleaned.select('Primary Type').distinct()
            # Mostro i risultati
            unique_values.show(truncate=False)
            #conto i crimini per categoria
            crime_counts = self.df_cleaned.groupBy('Primary Type').agg(count("*").alias("Count")).orderBy("Count", ascending=False)
            crime_counts.write.mode("overwrite").parquet(parquet_pathplot)
        
        # Converto in Pandas per visualizzarli in streamlit
        crime_counts_pd = crime_counts.toPandas()
        # Riordino il DataFrame Pandas per essere sicuro dell'ordine decrescente
        crime_counts_pd = crime_counts_pd.sort_values(by="Count", ascending=False)

        #st.header("📊 Crime Analysis by Type")

        # Titolo per il primo grafico
        st.subheader("🔍 Distribution of Crimes by Primary Type")

        # Creo il grafico a barre
        plt.figure(figsize=(12, 8))
        plt.bar(crime_counts_pd['Primary Type'], crime_counts_pd["Count"], color="skyblue")
        plt.xticks(rotation=90, fontsize=10)
        plt.title("Number of crimes by Primary Type", fontsize=16)
        plt.xlabel("Type of crime", fontsize=12)
        plt.ylabel("Number of crimes", fontsize=12)
        plt.tight_layout()
        
        st.pyplot(plt)
        # Aggiungo un divisore per separare i contenuti
        st.divider()
    

    def homeView(self):
        # Se esiste, carica il DataFrame da Parquet
        #st.write("DataFrame caricato da Parquet:", df_home)
        st.dataframe(self.df_cleaned)
        st.subheader("Null counts for cols in DataFrame:")
        st.dataframe(self.null_counts.toPandas())
        
        st.subheader("Distribution of data types in DataFrame")
        types_of_data = self.df_cleaned.dtypes
        type_counts = Counter([dtype for _, dtype in types_of_data])

        labels = list(type_counts.keys())
        sizes = list(type_counts.values())

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors,textprops={'fontsize': 5})
    
        st.pyplot(fig)
        
        
    def show_map(self):
        if os.path.exists("map.html"):
            with open("map.html", "r", encoding="utf-8") as f:
                map_html = f.read()
            st.components.v1.html(map_html, height=600)
        else:
            #inizializzo la mappa
            m = folium.Map(location = [41.881832, -87.623177],zoom_start=6)
            #crime_counts_pd = self.crime_counts.toPandas()
            #creazione dei marker cluster
            marker_cluster = MarkerCluster().add_to(m)
            location_data = self.df_cleaned.groupBy("District","Primary Type").agg(
                avg("Latitude").alias("Latitude"), 
                avg("Longitude").alias("Longitude"), 
                count("*").alias("Count")
            ).cache()
            location_data.count() #forzo il calcolo per mantenere i dati in cache
            location_data = location_data.collect() #la collect va fatta dopo
            
            assembler = VectorAssembler(inputCols=["Count"], outputCol="features")
            
            data_for_clustering = self.df_cleaned.groupBy("District").agg(count("*").alias("Count")).cache()

            data_for_clustering = assembler.transform(data_for_clustering)
            
            kmeans = KMeans(k=3, seed=0, featuresCol="features", predictionCol="cluster")
            model = kmeans.fit(data_for_clustering)

            clustered_data = model.transform(data_for_clustering)

            district_clusters = {row["District"]: row["cluster"] for row in clustered_data.collect()}

            cluster_colors = ["green", "yellow", "red"]
            district_colors = {district: cluster_colors[cluster] for district, cluster in district_clusters.items()}
            

            def style_function(feature):
                district = feature["properties"].get("DIST_NUM")
                if district:
                    color = district_colors.get(int(district), (0.5, 0.5, 0.5, 1.0))  # Default
                else:
                    color = (0.5, 0.5, 0.5, 1.0)  # Default
                try:
                    color_hex = to_hex(color)
                except ValueError:
                    raise ValueError(f"Colore RGBA non valido per il distretto {district}: {color}")
                return {
                    'fillColor': color_hex, # colore di riempimento
                    'color': "black",  # colore del bordo
                    'weight': 2, # larghezza del bordo
                    'fillOpacity': 0.7 # opacità del riempimento
                }

            # Carica i dati GeoJSON
            with open("output.geojson", 'r') as f:
                geojson_data = json.load(f)
            
            # Aggiungi i dati GeoJSON alla mappa
            folium.GeoJson(
                geojson_data,
                style_function=style_function,
                tooltip=folium.GeoJsonTooltip(fields=["DIST_NUM"]),
            ).add_to(m)
            
            # Aggiungi i marker alla mappa
            for row in location_data:
                folium.Marker(
                    location=[row["Latitude"], row["Longitude"]],
                    popup=folium.Popup(f"Crimine: {row['Primary Type']}, Numero di crimini: {row['Count']}", max_width=300),
                    icon=folium.Icon(color="blue", icon="info-sign")
                ).add_to(marker_cluster)
            
            # Salva la mappa su un file HTML
            m.save("map.html")
            print("Mappa salvata come 'map.html'")
            map_html = m._repr_html_()  # Genera l'HTML della mappa
            st.components.v1.html(map_html, width=1000, height=600)

    def show_district(self):
        parquet_home = "dataset_files/district_counts.parquet"
        if file_exists(self,parquet_home):
            district_counts = self.spark.read.parquet(parquet_home)
        else:
            district_counts = self.df_cleaned.groupBy('District').count()
            #district_counts.rename({'count(1)': 'Count'})
            district_counts.write.mode("overwrite").parquet(parquet_home)
        district_counts= district_counts.select("District", "count").toPandas()
        st.subheader("🌃 Distribution of crimes by District")
        district_counts["District"] = district_counts["District"].astype(int)  # Assicuriamoci che sia numerico
        district_counts = district_counts.sort_values("District", ascending=True)  # Ordiniamo numericamente
        
        # Creiamo una colormap per un effetto sfumato
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(district_counts)))

        # Creazione del grafico
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(district_counts["District"], district_counts["count"], color=colors)

        # Aggiungiamo i valori sopra le barre
        for bar, count in zip(bars, district_counts["count"]):
            ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height()/2, str(count), va="center", fontsize=9)

        ax.set_xlabel("Number of crimes")
        ax.set_ylabel("District", fontsize=12)
        ax.tick_params(axis='y', labelsize=10)
        ax.set_yticks(district_counts["District"])  # Imposta tutti i valori dell'asse Y
        plt.yticks(rotation=0) 
        ax.grid(axis="x", linestyle="--", alpha=0.7)

        st.pyplot(fig)
        #st.dataframe(district_counts)
        
    
    def show_season(self):    
        parquet_season= "dataset_files\season.parquet"

        if file_exists(self,parquet_season):
            season_counts = self.spark.read.parquet(parquet_season)
            season_counts_pd = season_counts.toPandas() 
            season_counts_pd = season_counts_pd.convert_dtypes()
        else:
            season_counts = (
                self.df_season.groupBy("year", "season")
                .count()
                .groupBy("year")
                .pivot("season")
                .sum("count")
                .fillna(0)
                .cache() 
            )
            
            season_counts_pd = season_counts.toPandas()
            season_counts_pd = season_counts_pd.convert_dtypes()
            season_counts.write.mode("overwrite").parquet(parquet_season)


        season_counts_pd = season_counts_pd.set_index("year")# Imposta "year" come indice
        season_counts_pd.index = season_counts_pd.index.astype(str)
        season_counts_pd = season_counts_pd.sort_index()
        
        fig, ax = plt.subplots(figsize=(12, 8))  # Imposta dimensioni più grandi

        # Colori per le stagioni
        season_colors = {"Winter": "blue", "Spring": "green", "Summer": "orange", "Autumn": "brown"}

        bottoms = [0] * len(season_counts_pd)  # Lista per tracciare l'altezza cumulativa
        for season, color in season_colors.items():
            if season in season_counts_pd.columns:
                plt.bar(
                    season_counts_pd.index,
                    season_counts_pd[season],
                    label=season,
                    color=color,
                    bottom=bottoms,
                )
                bottoms = [bottoms[i] + season_counts_pd[season].iloc[i] for i in range(len(bottoms))]
        
        # Personalizza il grafico
        ax.set_title("Occurrences by Season in Each Year", fontsize=16)
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Number of Occurrences", fontsize=12)
        ax.legend(title="Season")
        ax.set_xticklabels(season_counts_pd.index, rotation=45, ha="right")  # Ruota le etichette
        plt.tight_layout()  # Migliora la spaziatura

        # Streamlit UI
        st.subheader("📊 Occurrences by Season in Each Year")  
        st.write("This graph shows the number of occurrences broken down by season from 2003 to 2020.")  
        st.pyplot(fig)  # Mostra il grafico con Streamlit
        st.divider()

    #def show_avg(self): 
    #    parquet_avg= "dataset_files\\avg.parquet"
    #    if file_exists(self,parquet_avg):
    #        #avg_daily_crimes = self.spark.read.parquet(parquet_avg)
    #        avg_daily_crimes = self.spark.read.parquet(parquet_avg).orderBy("month")
    #        st.dataframe(
    #        avg_daily_crimes.toPandas()
    #        .rename(columns={"month": "Month", "avg_daily_crimes": "Average monthly crimes"})
    #        .style.format({"Average monthly crimes": "{:.2f}"})
    #    )
    #    else:
    #        daily_crimes = self.df_cleaned.groupBy(
    #            col("month").alias("month"),
    #            col("day").alias("day")
    #        ).count()
    #       
    #        avg_daily_crimes = daily_crimes.groupBy("month").agg(avg("count").alias("avg_daily_crimes"))
    #        result = avg_daily_crimes.orderBy("month").collect()
            
            # Prepara i dati per Streamlit
    #        data = [{"Month": row['month'], "Average monthly crimes": row['avg_daily_crimes']} for row in result]
    #       st.write("Average monthly crimes") 
    #        st.table(data)
    #        avg_daily_crimes.write.mode("overwrite").parquet(parquet_avg)


    def show_perc_arrests(self):
        perc_arrests_path = "dataset_files\\perc_arrests.parquet" 
        if(file_exists(self,perc_arrests_path)):
            perc_arrests = self.spark.read.parquet(perc_arrests_path)
        else:
            perc_arrests = self.df_cleaned.select(
                (sum((col("Arrest") == 1).cast("int")) * 100 / count("*")).alias("perc_arrests")
            )
            perc_arrests.write.mode("overwrite").parquet(perc_arrests_path)

        st.subheader(f"⛓️ Percentage of arrests: {perc_arrests.collect()[0][0] :.2f}%") 
        #st.table(perc_arrests.toPandas())


    def show_common_crimes_location(self):
        parquet_location = "dataset_files\location.parquet"
        if file_exists(self,parquet_location):
            common_crimes_location = self.spark.read.parquet(parquet_location)
        else:
            common_crimes_location = self.df_cleaned.groupBy("Location Description").agg(count("*").alias("count")).orderBy("count", ascending=False).limit(10)
            common_crimes_location.write.mode("overwrite").parquet(parquet_location)
        st.subheader("🌍 First 10 location for number of crimes")
        common_crimes_location = common_crimes_location.toPandas()
        #st.table(common_crimes_location.toPandas())
        fig, ax = plt.subplots(figsize=(10, 6))
    
        # Creiamo il grafico a barre orizzontali con numerazione
        bars = ax.barh(range(len(common_crimes_location)), common_crimes_location["count"], color="purple")
        ax.set_yticks(range(len(common_crimes_location)))
        ax.set_yticklabels(common_crimes_location["Location Description"])

        # Aggiungiamo numeri accanto a ogni barra
        for i, bar in enumerate(bars):
            ax.text(bar.get_width() + 5000, bar.get_y() + bar.get_height()/2, f"#{i+1}", va="center", fontsize=12, fontweight="bold")

        ax.set_xlabel("Number of crimes")

        ax.invert_yaxis()  # Per mettere il primo posto in cima
        plt.grid(axis="x", linestyle="--", alpha=0.7)

        st.pyplot(fig)



    def show_area_violent_crimes(self):
        violent_crimes_path = "dataset_files\\violent_crimes.parquet"
        if file_exists(self,violent_crimes_path):
            crimes_for_area = self.spark.read.parquet(violent_crimes_path)    
        else:
            violent_crimes = self.df_cleaned.filter(col("Primary Type").isin(["HOMICIDE", "ASSAULT", "ROBBERY"]))
            # Conta i crimini violenti per quartiere
            crimes_for_area = violent_crimes.groupBy("Community Area").count().orderBy(col("count").desc()).limit(10)
            crimes_for_area.write.mode("overwrite").parquet(violent_crimes_path)

        st.subheader("🔪First 10 area for violent crimes")
        st.table(crimes_for_area.toPandas())


    def show_moving_average(self):
        moving_average_path = "dataset_files\\moving_average.parquet"
        if file_exists(self,moving_average_path):
            moving_average = self.spark.read.parquet(moving_average_path)
        else:
            moving_average = self.df_cleaned.groupBy("year","month").agg(count("*").alias("Crimes_count"))
            windowSpec = Window.partitionBy("year", "month").orderBy("year").rowsBetween(-2, 0)
            moving_average = moving_average.withColumn("moving_average", avg("Crimes_count").over(windowSpec))
            moving_average.write.mode("overwrite").parquet(moving_average_path)
        st.subheader("📅 Moving average of crimes per month")
        df = moving_average.toPandas()
        df["Year-Month"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str))

        df = df.sort_values("Year-Month")

        plt.figure(figsize=(12, 6))
        sns.lineplot(x=df["Year-Month"], y=df["moving_average"], marker="o", label="3-Month Moving Avg", color="blue", linewidth=2.5)

        plt.xlabel("Year-Month")
        plt.ylabel("Number of Crimes")
        plt.title("Moving Average of Crimes Per Month")
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)

        st.pyplot(plt)

    def hist_arrests(self,pdf):
            st.title("📊 Histogram of Arrests")
            pdf = pdf.withColumn("Correct", (pdf["prediction"] == pdf["Arrest"]).cast("integer"))

            results_count = pdf.groupBy("Correct").count().toPandas()

            labels = ["Incorrect Predictions", "Correct Predictions"]
            counts = [results_count.loc[results_count["Correct"] == 0, "count"].values[0],
                    results_count.loc[results_count["Correct"] == 1, "count"].values[0]]

            # Creazione istogramma
            plt.figure(figsize=(6, 4))
            plt.bar(labels, counts, color=["red", "green"])
            plt.xlabel("Prediction Outcome")
            plt.ylabel("Count")
            plt.title("Correct vs Incorrect Predictions")
            plt.xticks(rotation=0)

            st.pyplot(plt)
    
    def logisticregression(self):
        #self.df_cleaned = self.df_cleaned.withColumn("Arrest", self.df_cleaned["Arrest"].cast("integer"))
        st.title("🚆Logistic Regression on Arrests")
        model_path = "dataset_files\\logistic_regression_model.parquet"
        if file_exists(self,model_path):
            predictions = self.spark.read.parquet(model_path)
            #st.write("✅ Model loaded from HDFS!")
            roc_auc = self.spark.read.parquet("dataset_files\\roc_auc.parquet")
            roc_auc = roc_auc.collect()[0][0]
        else:
            exclude_from_data = ["Arrest", "ID", "Domestic","Case Number", "dayofweek","Updated On" , "Longitude", "Latitude", "day", "month", "Community Area"]
            df_mod = self.df_cleaned.drop(*[col[0] for col in self.df_cleaned.dtypes if col[1] in ('timestamp','date')])

            Xs= [item for item in df_mod.columns if item not in exclude_from_data]

            to_encode= [col[0] for col in df_mod.select(*Xs).dtypes if col[1]=='string']
            numerical_cols= [col[0] for col in df_mod.select(*Xs).dtypes if col[1]!='string']
            pipe_stages= []

            sindexer= StringIndexer(inputCols= to_encode, 
                            outputCols= ["indexed_{}".format(item) for item in to_encode],
                            handleInvalid='keep',
                            stringOrderType='frequencyDesc')
            
            pipe_stages += [sindexer]


            assembler= VectorAssembler(inputCols= ["indexed_{}".format(item) for item in to_encode] + [item for item in numerical_cols],
                            outputCol= "feats",
                            handleInvalid="keep")
            
            pipe_stages += [assembler]

            ss= StandardScaler(inputCol="feats",
                            outputCol="features",
                            withMean= False,
                            withStd=True)
            pipe_stages += [ss]

            label = "Arrest"

            # Creazione della pipeline
            pipeline = Pipeline(stages=pipe_stages)

            # Training del modello
            df_mod = pipeline.transform(df_mod)

            # Divisione Train-Test
            train_data, test_data = df_mod.randomSplit([0.8, 0.2], seed=42)
            
            # Definizione del modello Logistic Regression
            lr = LogisticRegression(featuresCol="features", labelCol=label, maxIter=100, regParam=0.1)
            #paramGrid = (ParamGridBuilder()
            # .addGrid(lr.maxIter, [10, 20, 50])  # Numero massimo di iterazioni
            # .addGrid(lr.regParam, [0.01, 0.1, 1.0])  # Regolarizzazione L2
            # .build())

            # Definizione dell'evaluator per AUC
            #evaluator = BinaryClassificationEvaluator(labelCol="Arrest", metricName="areaUnderROC")

            # Configurazione della Cross Validation
            #crossval = CrossValidator(
            #    estimator=lr,
            #    estimatorParamMaps=paramGrid,
            #    evaluator=evaluator,
            #    numFolds=5,  # Numero di fold per la validazione incrociata
            #    parallelism=2  # Parallelismo per velocizzare il processo
            #)

            # Esegui la Cross Validation e trova il miglior modello
            #cv_model = crossval.fit(train_data)

            # Estrai il miglior modello
            #best_model = cv_model.bestModel

            # Fai le predizioni sul test set
            #predictions = best_model.transform(test_data)

            # Valutazione finale del modello sul test set
            #roc_auc = evaluator.evaluate(predictions)

            model = lr.fit(train_data)

            predictions = model.transform(test_data)


            # Salva il modello in HDFS
            predictions.write.mode("overwrite").parquet(model_path)
            #st.write("✅ Model trained and saved to HDFS!")

            evaluator = BinaryClassificationEvaluator(labelCol="Arrest", metricName="areaUnderROC")
            roc_auc = evaluator.evaluate(predictions)
            
            df = self.spark.createDataFrame([(roc_auc,)], ["roc_auc"])
            df.write.mode("overwrite").parquet("dataset_files\\roc_auc.parquet")

        # Converti le predizioni in Pandas per visualizzazione
        pdf = predictions.select("ID", "Case Number","Description","Primary Type", "Location Description","District", "Arrest", "prediction")

        # Mostra la tabella con le previsioni
        st.subheader("🔍 Model Predictions")
        st.dataframe(pdf.toPandas(), use_container_width=True)

        self.hist_arrests(pdf)

        # Mostra il punteggio ROC AUC
        st.subheader(f"📊 ROC AUC Score: {roc_auc:.4f}") 
        
        # 🔹 **Curva ROC**
        #training_summary = model.stages[-1].summary  
        #roc_df = training_summary.roc.toPandas()

        #fig, ax = plt.subplots()
        #ax.plot([0, 1], [0, 1], "r--")  
        #ax.plot(roc_df["FPR"], roc_df["TPR"], label="Model ROC Curve")
        #ax.set_xlabel("False Positive Rate")
        #ax.set_ylabel("True Positive Rate")
        #ax.set_title("ROC Curve (PySpark)")
        #ax.legend()
        #st.pyplot(fig)"""


    def random_forest_arrests(self):
        path_prediction ="dataset_files\\random_forest_model.parquet"
        st.title("🌳Random Forest on Arrests")
        if file_exists(self,path_prediction):
            predictions = self.spark.read.parquet(path_prediction)
            auc = self.spark.read.parquet("dataset_files\\auc.parquet")
            auc = auc.collect()[0][0]
        else:
            df_mod = self.df_cleaned.drop(*[col[0] for col in self.df_cleaned.dtypes if col[1] in ('timestamp','date')])
            print(df_mod.columns)
            #escludiamo la colonna target e la colonna ID
            exclude_from_data = ["Arrest", "ID", "Case Number", "Domestic","year", "month", "dayofweek", "day", "hour","Updated On" ]
            Xs= [item for item in df_mod.columns if item not in exclude_from_data]

            to_encode= [col[0] for col in df_mod.select(*Xs).dtypes if col[1]=='string']
            numerical_cols= [col[0] for col in df_mod.select(*Xs).dtypes if col[1]!='string']

            pipe_stages= []

            sindexer= StringIndexer(inputCols= to_encode, 
                            outputCols= ["indexed_{}".format(item) for item in to_encode],
                            handleInvalid='keep',
                            stringOrderType='frequencyDesc')
            
            pipe_stages += [sindexer]

            assembler= VectorAssembler(inputCols= ["indexed_{}".format(item) for item in to_encode] + [item for item in numerical_cols],
                            outputCol= "feats",
                            handleInvalid="keep")
            
            pipe_stages += [assembler]

            ss= StandardScaler(inputCol="feats",
                            outputCol="features",
                            withMean= False,
                            withStd=True)
            pipe_stages += [ss]

            label = "Arrest"

            # creo la pipeline per automatizzare il processo di addestramento
            pipeline = Pipeline(stages=pipe_stages)
            df_mod=pipeline.transform(df_mod)

            #suddivisione del dataset in train e test
            train_data, test_data = df_mod.randomSplit([0.7, 0.3],seed=42)
            
            rf = RandomForestClassifier(labelCol=label, featuresCol="features", numTrees=20,maxDepth=10,seed=42)

            rfc_model = rf.fit(train_data)
            predictions = rfc_model.transform(test_data)
            predictions.write.mode("overwrite").parquet("dataset_files\\random_forest_model.parquet")
            evaluator = BinaryClassificationEvaluator(labelCol=label, metricName="areaUnderROC")
            auc = evaluator.evaluate(predictions)
            schema = StructType([StructField("value", FloatType(), True)])
            df = self.spark.createDataFrame([(auc,)], schema)
            df.write.mode("overwrite").parquet("dataset_files\\auc.parquet")
            

        #qua possiamo aggiungere la cross validation
        #paramGrid = ParamGridBuilder() \
        #    .addGrid(rf.numTrees, [10, 15, 20]) \
        #    .addGrid(rf.maxDepth, [5, 10]) \
        #    .build()

        

        #crossval = CrossValidator(
        #    estimator=rf,
        #    estimatorParamMaps=paramGrid,
        #    evaluator=BinaryClassificationEvaluator(labelCol="Arrest"),
        #    numFolds=5
        #)
        
        #cv_model = crossval.fit(train_data)
        #predictions = cv_model.bestModel.transform(test_data)

        #rfc_model = crossval.fit(train_data)
        #best_model = rfc_model.bestModel

        #print(f"Miglior numero di alberi: {best_model.getNumTrees}")
        #print(f"Miglior profondità: {best_model.getMaxDepth}")

        #print(f"Feature names: {assembler.getInputCols()}")

        #predictions = best_model.transform(test_data)

        #print(f"Area Under ROC Curve (AUC) = {auc}")

        #feature_importances = rfc_model.featureImportances
        #print(feature_importances)

        ##rmse_evaluator = RegressionEvaluator(labelCol=label, predictionCol="prediction", metricName="rmse")
        #rmse = rmse_evaluator.evaluate(predictions)
        #print(f"Root Mean Squared Error (RMSE) = {rmse}")

        #df_mod.groupBy("Arrest").count().show()
        df_result = predictions.select("ID", "Case Number", "Description" ,"Primary Type", "Location Description","District", "Arrest", "prediction")
        
        st.subheader("🔍 Model Predictions")
        st.dataframe(df_result.toPandas(), use_container_width=True)
        self.hist_arrests(df_result)

        # Mostra il punteggio ROC AUC
        st.subheader(f"📊 ROC AUC Score: {auc:.4f}") 


    def grandient_boosting_crimines(self):
        path_gbc = "dataset_files\\gbc_model.parquet"
        st.title("⚡Gradient Boosting on Crimes Count")
        if file_exists(self,path_gbc):
            predictions = self.spark.read.parquet(path_gbc)
            rmse = self.spark.read.parquet("dataset_files\\rmse.parquet")
            rmse = rmse.collect()[0][0]
        else:
            df_mod=  self.df_cleaned.groupBy("District", "year","day","month" ).count().orderBy("year", "month", "day")
            #df_mod = df_mod.withColumn("isWeekend", when(df_mod["dayofweek"].isin([6, 7]), 1).otherwise(0))

            df_mod = df_mod.withColumnRenamed("count","Crimes_count")
            df_mod.show()
            df_mod.printSchema()
            exclude_from_data = ["Crimes_count",]
            Xs= [item for item in df_mod.columns if item not in exclude_from_data]

            #to_encode= [col[0] for col in df_mod.select(*Xs).dtypes if col[1]=='string']
            numerical_cols= [col[0] for col in df_mod.select(*Xs).dtypes if col[1]!='string']

            pipe_stages= []

            #sindexer= StringIndexer(inputCols= to_encode, 
                            #outputCols= ["indexed_{}".format(item) for item in to_encode],
                            #handleInvalid='keep',
                            #stringOrderType='frequencyDesc')
            
            #pipe_stages += [sindexer]

            assembler= VectorAssembler(inputCols= [item for item in numerical_cols] ,
                            outputCol= "feats",
                            handleInvalid="keep")
            
            pipe_stages += [assembler]

            ss= MinMaxScaler(inputCol="feats",
                            outputCol="features",
                            )
            pipe_stages += [ss]

            label = "Crimes_count"

            pipeline = Pipeline(stages=pipe_stages)
            df_mod=pipeline.transform(df_mod)

            train_data, test_data = df_mod.randomSplit([0.7, 0.3],seed=42)
            
            gbt = GBTRegressor(labelCol="Crimes_count", 
                    featuresCol="features", 
                    maxIter=100,  # Numero di iterazioni (alberi)
                    maxDepth=5,  # Profondità massima dell'albero
                    stepSize=0.05, # Learning rate (valori più bassi = più precisi ma più lenti)
                    seed=42)

            gbt_model = gbt.fit(train_data)
            predictions = gbt_model.transform(test_data)
            predictions.write.mode("overwrite").parquet("dataset_files\gbc_model.parquet")
            #feature_importances = gbt_model.featureImportances
            #print(feature_importances)

            evaluator = RegressionEvaluator(labelCol=label, predictionCol="prediction", metricName="rmse")
            rmse = evaluator.evaluate(predictions)

            schema = StructType([StructField("value", FloatType(), True)])
            df = self.spark.createDataFrame([(rmse,)], schema)
            df.write.mode("overwrite").parquet("dataset_files\\rmse.parquet")

        #print(predictions)
        st.subheader("🔍 Model Predictions")
        st.dataframe(predictions.select("District", "year","month","day", "Crimes_count","prediction",).toPandas())
        
        st.subheader(f"📊 RMSE: {rmse:.4f}") 

    def correlation_matrix(self):
        self.df_cleaned = self.df_cleaned.drop("ID")
        st.title("Correlation Matrix")
        to_encode= [col[0] for col in self.df_cleaned.select().dtypes if col[1]=='string']
        numerical_cols = [col[0] for col in self.df_cleaned.dtypes if col[1] in ('int', 'double')]

        sindexer= StringIndexer(inputCols= to_encode, 
                            outputCols= ["indexed_{}".format(item) for item in to_encode],
                            handleInvalid='keep',
                            stringOrderType='frequencyDesc')

        vector_col = "features"
        feature_cols = numerical_cols + [f"indexed_{col}" for col in to_encode]
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        
        pipeline = Pipeline(stages=[sindexer] + [assembler])
        df_transformed = pipeline.fit(self.df_cleaned).transform(self.df_cleaned)

        correlation_matrix = Correlation.corr(df_transformed, vector_col, "pearson").head()[0].toArray()

        correlation_df = pd.DataFrame(correlation_matrix, index=feature_cols, columns=feature_cols)

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_df, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Matrice di Correlazione")

        st.pyplot(plt)

class Frontend:
    def __init__(self, dataset:Dataset):
        self.dataset = dataset

    def frontend(self):
        with st.sidebar:
            selected = option_menu(
                menu_title="Menu",
                options=["Home", "Histograms", "Map", "More", "ML Models"],
                icons=["house", "bar-chart", "map", "table", "rocket"],
                default_index=0,
            )

        if selected == "Home":
            st.title("Crimes from 2001 to present in Chicago")
            self.dataset.homeView()
                

        elif selected == "Histograms":
            st.title("Histograms")
            self.dataset.showPlotPrimaryType()
            self.dataset.show_season()
            self.dataset.show_district()
            #self.dataset.show_avg()
            
            
        elif selected == "Map":
            st.title("Map")
            #st.write("Visualizza la mappa interattiva qui.")
            self.dataset.show_map()

        elif selected == "More":
            st.title("More")
            self.dataset.showCriticalHour()
            self.dataset.show_perc_arrests()
            self.dataset.show_common_crimes_location()
            self.dataset.show_area_violent_crimes()
            self.dataset.show_moving_average()
            
        elif selected == "ML Models":
            st.title("ML Models")
            self.dataset.correlation_matrix()
            #st.write("Risultati del modello di machine learning.")
            self.dataset.logisticregression()
            self.dataset.random_forest_arrests()
            self.dataset.grandient_boosting_crimines()


def main():
    dataset = Dataset()
    frontend = Frontend(dataset)
    frontend.frontend()

if __name__ == "__main__":
    main()