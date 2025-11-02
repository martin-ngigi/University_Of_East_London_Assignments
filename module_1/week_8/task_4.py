import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import openpyxl

warnings.filterwarnings('ignore')


def load_property_data(filepath):
    """Load property price data with proper column names."""

    # Define column names based on Land Registry specification
    columns = [
        'Transaction_ID',
        'Price',
        'Date',
        'Postcode',
        'Property_Type',
        'Old_New',
        'Duration',
        'PAON',
        'SAON',
        'Street',
        'Locality',
        'Town_City',
        'District',
        'County',
        'PPD_Category',
        'Record_Status'
    ]

    df = pd.read_csv(filepath, header=None, names=columns)
    return df


def process_property_data(df):
    """Process property price data according to Task 4 requirements."""

    print("=== Initial Data Info ===")
    print(f"Total records: {len(df)}")
    print(f"\nDate range: {df['Date'].min()} to {df['Date'].max()}")

    # Convert Date column to datetime (handles various formats)
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=False)

    # Sort by Date
    df = df.sort_values('Date', ascending=True)

    # Find the latest complete month
    latest_date = df['Date'].max()
    print(f"\nLatest date in data: {latest_date}")

    # Get the last complete month (month before the latest month if latest is incomplete)
    # Check if we have transactions throughout the latest month
    latest_year = latest_date.year
    latest_month = latest_date.month

    # Filter for the latest complete month
    # Assuming the latest month in the data is complete
    df_filtered = df[(df['Date'].dt.year == latest_year) &
                     (df['Date'].dt.month == latest_month)]

    print(f"\nFiltered to latest complete month: {latest_year}-{latest_month:02d}")
    print(f"Records after date filter: {len(df_filtered)}")

    # Check Record_Status column
    print(f"\nRecord Status counts:")
    print(df_filtered['Record_Status'].value_counts())

    # Keep only 'A' records
    df_filtered = df_filtered[df_filtered['Record_Status'] == 'A']
    print(f"Records after removing C/D status: {len(df_filtered)}")

    # Filter for property types D, F, S, T only (exclude O)
    valid_property_types = ['D', 'F', 'S', 'T']
    df_filtered = df_filtered[df_filtered['Property_Type'].isin(valid_property_types)]
    print(f"Records after filtering property types (D, F, S, T): {len(df_filtered)}")

    print(f"\nProperty Type distribution:")
    print(df_filtered['Property_Type'].value_counts())

    return df_filtered


def create_pivot_by_district(df):
    """Create pivot table counting transactions by District and Property Type."""

    # Create pivot table
    pivot = pd.pivot_table(
        df,
        values='Transaction_ID',
        index='District',
        columns='Property_Type',
        aggfunc='count',
        fill_value=0
    )

    # Ensure all property types are present
    for prop_type in ['D', 'F', 'S', 'T']:
        if prop_type not in pivot.columns:
            pivot[prop_type] = 0

    # Reorder columns
    pivot = pivot[['D', 'F', 'S', 'T']]

    # Add total column
    pivot['Total'] = pivot.sum(axis=1)

    # Reset index to make District a column
    pivot = pivot.reset_index()

    print("\n=== Pivot Table by District ===")
    print(f"Shape: {pivot.shape}")
    print(f"\nFirst 10 rows:")
    print(pivot.head(10))

    return pivot


def match_with_lookup(pivot_df, lookup_df):
    """Match district pivot with lookup table and handle mismatches."""

    # Clean district names for matching
    pivot_df['District_Clean'] = pivot_df['District'].str.strip().str.upper()
    lookup_df['LAD_Name_Clean'] = lookup_df['LAD_Name'].str.strip().str.upper()

    # Merge with lookup
    merged = pivot_df.merge(
        lookup_df[['LAD_Name', 'LAD_Name_Clean', 'Region_Code', 'Region_Name']],
        left_on='District_Clean',
        right_on='LAD_Name_Clean',
        how='left'
    )

    # Check for unmatched districts
    unmatched = merged[merged['Region_Name'].isna()]
    if len(unmatched) > 0:
        print("\n⚠ WARNING: Unmatched Districts:")
        print(unmatched[['District', 'D', 'F', 'S', 'T', 'Total']])
        print("\nThese districts need manual adjustment in the lookup table.")

    # Remove unmatched rows (districts with no sales or name mismatches)
    matched = merged[merged['Region_Name'].notna()].copy()

    # Select final columns
    result = matched[[
        'District',
        'LAD_Name',
        'Region_Code',
        'Region_Name',
        'D', 'F', 'S', 'T',
        'Total'
    ]]

    print(f"\n=== Matched Results ===")
    print(f"Matched districts: {len(result)}")
    print(f"Unmatched districts: {len(unmatched)}")

    return result


def create_regional_summary(district_df):
    """Create regional summary by summing property types."""

    regional_summary = district_df.groupby(['Region_Code', 'Region_Name']).agg({
        'D': 'sum',
        'F': 'sum',
        'S': 'sum',
        'T': 'sum',
        'Total': 'sum'
    }).reset_index()

    print("\n=== Regional Summary ===")
    print(regional_summary)

    return regional_summary


def main():
    """Main function to process property price data per Task 4."""

    print("=" * 60)
    print("TASK 4: Property Price Data Processing")
    print("=" * 60)

    # Load property price data
    print("\nLoading property price data...")
    property_df = load_property_data('module_1/week_8/pp-monthly-update-new-version.csv')

    # Process data (filter by date, status, property type)
    print("\nProcessing data...")
    processed_df = process_property_data(property_df)

    # Create pivot table by district and property type
    print("\nCreating pivot table...")
    pivot_district = create_pivot_by_district(processed_df)

    # Save pivot table
    pivot_district.to_csv('module_1/week_8/task_4_district_property_counts.csv', index=False)
    print("\n✓ Pivot table saved to: module_1/week_8/task_4_district_property_counts.csv")

    # Load lookup data (from Task 3 output)
    print("\nLoading lookup data...")
    try:
        lookup_df = pd.read_csv('module_1/week_8/census_dwelling_data_prepared.csv')

        # Match with lookup
        print("\nMatching with lookup data...")
        matched_df = match_with_lookup(pivot_district, lookup_df)

        # Save matched data
        matched_df.to_csv('module_1/week_8/task_4_district_property_counts_matched.csv', index=False)
        print("\n✓ Matched data saved to: module_1/week_8/task_4_district_property_counts_matched.csv")

        # Create regional summary
        print("\nCreating regional summary...")
        regional_df = create_regional_summary(matched_df)

        # Save regional summary
        regional_df.to_csv('module_1/week_8/task_4_regional_property_summary.csv', index=False)
        print("\n✓ Regional summary saved to: module_1/week_8/task_4_regional_property_summary.csv")

        # Create Excel file with multiple sheets
        print("\nCreating Excel output file...")
        with pd.ExcelWriter('module_1/week_8/property_analysis_output.xlsx', engine='openpyxl') as writer:
            pivot_district.to_excel(writer, sheet_name='District_Pivot', index=False)
            matched_df.to_excel(writer, sheet_name='District_Matched', index=False)
            regional_df.to_excel(writer, sheet_name='Regional_Summary', index=False)

        print("\n✓ Excel file saved to: property_analysis_output.xlsx")

    except FileNotFoundError:
        print("\n⚠ Warning: module_1/week_8/census_dwelling_data_prepared.csv not found.")
        print("Please run Task 3 first to generate the lookup file.")
        print("Saving pivot table only.")

    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()