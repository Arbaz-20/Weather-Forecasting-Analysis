import streamlit as st
from weatherApp import (load_data, melt_data, plot_temperature_timeline, 
                       plot_monthly_temperature_box, plot_kmeans_clusters, 
                       fit_kmeans_and_plot, plot_temperature_histogram, 
                       plot_yearly_mean_temperature, plot_monthly_temperature_history,plot_seasonal_mean_temperatures, predict_temperatures)

def main():
    st.title("Weather Forecasting Data Analysis")
    st.sidebar.title("Weather Predictor")
    option = st.sidebar.selectbox("Select an option",["Data Upload and Overview", "Data Visualization", "Predictive Analysis"])

    # Page 1: Data Upload and Overview
    if option == "Data Upload and Overview":
        st.header("Upload CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            if isinstance(df, str):
                if df == "ParserError":
                    st.error("Error parsing the file. Please ensure it is a valid CSV file.")
                else:
                    st.error(f"An error occurred: {df}")
            elif df is not None and not df.empty:
                st.write("Data Overview:")
                st.dataframe(df.head())
                st.write("Data Statistics:")
                st.write(df.describe())
            else:
                st.error("The uploaded file is empty or improperly formatted.")
    
    # Page 2: Data Visualization
    elif option == "Data Visualization":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            if isinstance(df, str):
                if df == "ParserError":
                    st.error("Error parsing the file. Please ensure it is a valid CSV file.")
                else:
                    st.error(f"An error occurred: {df}")
            elif df is not None and not df.empty:
                df1 = melt_data(df)
                st.subheader("Temperature Through Timeline")
                fig = plot_temperature_timeline(df1)
                st.plotly_chart(fig)
                
                st.subheader("Monthly Temperature Distribution")
                fig = plot_monthly_temperature_box(df1)
                st.plotly_chart(fig)
                
                st.subheader("KMeans Clustering Analysis")
                fig = plot_kmeans_clusters(df1)
                st.plotly_chart(fig)
                
                st.subheader("KMeans Clusters Visualization")
                fig = fit_kmeans_and_plot(df1)
                st.plotly_chart(fig)
                
                st.subheader("Temperature Histogram")
                fig = plot_temperature_histogram(df1)
                st.plotly_chart(fig)
                
                st.subheader("Yearly Mean Temperature")
                fig = plot_yearly_mean_temperature(df)
                st.plotly_chart(fig)
                
                st.subheader("Monthly Temperature History")
                fig = plot_monthly_temperature_history(df1)
                st.plotly_chart(fig)
                
                st.subheader("Seasonal Mean Temperatures")
                fig = plot_seasonal_mean_temperatures(df)
                st.plotly_chart(fig)
            else:
                st.error("The uploaded file is empty or improperly formatted.")
        else:
            st.info("Please upload a CSV file to proceed.")
    
    # Page 3: Predictive Analysis
    elif option == "Predictive Analysis":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            if isinstance(df, str):
                if df == "ParserError":
                    st.error("Error parsing the file. Please ensure it is a valid CSV file.")
                else:
                    st.error(f"An error occurred: {df}")
            elif df is not None and not df.empty:
                df1 = melt_data(df)
                st.subheader("Temperature Prediction")
                fig, r2 = predict_temperatures(df1)
                st.plotly_chart(fig)
                st.write(f"R² Score of the prediction model: {r2:.2f}")
            else:
                st.error("The uploaded file is empty or improperly formatted.")
        else:
            st.info("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()