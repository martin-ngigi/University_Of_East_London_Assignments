import pandas as pd
import numpy as np


def main():
    """Main function to prepare census dwelling data for Land Registry matching."""

    # Load the data files
    lookup_df = pd.read_csv('../../Local_Authority_District_to_Region_(December_2023)_Lookup_in_England.csv')
    census_df = pd.read_csv('../../RM205-2021-2.csv')

    # Display initial data structure
    print("=== Census Data Structure ===")
    print(census_df.head(10))
    print(f"\nShape: {census_df.shape}")
    print(f"\nColumns: {census_df.columns.tolist()}")

    # Clean column names (remove extra spaces)
    census_df.columns = census_df.columns.str.strip()
    lookup_df.columns = lookup_df.columns.str.strip()


    # Pivot the census data to wide format
    census_wide = census_df.pivot(
        index=['Lower tier local authorities Code', 'Lower tier local authorities'],
        columns='Number of household spaces in shared dwellings (3 categories)',
        values='Observation'
    ).reset_index()

    # Rename columns for clarity
    census_wide.columns.name = None
    census_wide.columns = [
        'LAD_Code',
        'LAD_Name',
        'Shared_Two_Spaces',
        'Shared_Three_Plus_Spaces',
        'Unshared_Dwelling'
    ]

    # Calculate total dwellings and dwelling type aggregations
    census_wide['Total_Dwellings'] = (
            census_wide['Shared_Two_Spaces'] +
            census_wide['Shared_Three_Plus_Spaces'] +
            census_wide['Unshared_Dwelling']
    )

    # Aggregate shared dwellings
    census_wide['Shared_Dwellings'] = (
            census_wide['Shared_Two_Spaces'] +
            census_wide['Shared_Three_Plus_Spaces']
    )

    # Merge with region lookup
    final_df = census_wide.merge(
        lookup_df[['LAD23CD', 'RGN23CD', 'RGN23NM']],
        left_on='LAD_Code',
        right_on='LAD23CD',
        how='left'
    )

    # Reorder and select relevant columns
    final_df = final_df[[
        'LAD_Code',
        'LAD_Name',
        'RGN23CD',
        'RGN23NM',
        'Unshared_Dwelling',
        'Shared_Dwellings',
        'Shared_Two_Spaces',
        'Shared_Three_Plus_Spaces',
        'Total_Dwellings'
    ]]

    # Rename for consistency with Land Registry conventions
    final_df.columns = [
        'LAD_Code',
        'LAD_Name',
        'Region_Code',
        'Region_Name',
        'Unshared_Dwellings',
        'Shared_Dwellings',
        'Shared_Two_Spaces',
        'Shared_Three_Plus_Spaces',
        'Total_Dwellings'
    ]

    # Display results
    print("\n=== Processed Census Dwelling Data ===")
    print(final_df.head(10))
    print(f"\nShape: {final_df.shape}")
    print(f"\nSummary Statistics:")
    print(final_df[['Unshared_Dwellings', 'Shared_Dwellings', 'Total_Dwellings']].describe())

    # Save to CSV
    output_file = '../../census_dwelling_data_prepared.csv'
    final_df.to_csv(output_file, index=False)
    print(f"\n✓ Data saved to: {output_file}")

    # Regional aggregation (optional)
    regional_summary = final_df.groupby(['Region_Code', 'Region_Name']).agg({
        'Unshared_Dwellings': 'sum',
        'Shared_Dwellings': 'sum',
        'Total_Dwellings': 'sum'
    }).reset_index()

    print("\n=== Regional Summary ===")
    print(regional_summary)

    # Save regional summary
    regional_output = 'census_dwelling_regional_summary.csv'
    regional_summary.to_csv(regional_output, index=False)
    print(f"\n✓ Regional summary saved to: {regional_output}")

# Entry point check
if __name__ == "__main__":
    main()