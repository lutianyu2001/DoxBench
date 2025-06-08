import os
import exifread
import folium
from http.server import HTTPServer, SimpleHTTPRequestHandler
import webbrowser
import argparse

def create_popup_html(file_path, address):
    """Create popup HTML with file info"""
    filename = os.path.basename(file_path) if file_path else "Unknown"
    
    html = f"""
    <div style="width: 300px;">
        <h4>{filename}</h4>
        <p><strong>Address:</strong> {address}</p>
        <p><strong>File:</strong> {file_path}</p>
    </div>
    """
    return html

def read_csv_data(csv_file):
    """Read location data from CSV file with specific format:
    Line 1: File path
    Line 2: Address
    Line 3: Latitude
    Line 4: Longitude
    """
    locations = []
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            # Process 4 lines at a time
            i = 0
            while i < len(lines) and i + 3 < len(lines):
                try:
                    file_path = lines[i].strip()
                    address = lines[i+1].strip()
                    lat = float(lines[i+2].strip())
                    lon = float(lines[i+3].strip())
                    
                    if lat != 0 and lon != 0:
                        locations.append((file_path, address, lat, lon))
                except (ValueError, TypeError) as e:
                    print(f"Error processing CSV data at line {i+1}: {e}")
                
                i += 4
    except Exception as e:
        print(f"Error reading CSV file: {e}")
    
    return locations

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Visualize location data from CSV file')
    parser.add_argument('--csv', type=str, required=True, help='Path to the CSV file containing location data')
    parser.add_argument('--output', type=str, default='location_map.html', help='Output map filename')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    csv_file = args.csv
    map_filename = args.output
    
    print(f"Reading location data from: {csv_file}")
    locations = read_csv_data(csv_file)
    
    if not locations:
        print('No valid location data found in CSV file.')
        return
    
    print(f"Found {len(locations)} locations with coordinates")
    
    # Create map
    print("Generating map...")
    m = folium.Map(location=[20, 0], zoom_start=2)
    
    # Add markers for each location
    for file_path, address, lat, lon in locations:
        popup_html = create_popup_html(file_path, address)
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=320)
        ).add_to(m)
    
    # Save map
    m.save(map_filename)
    print(f'Map saved as: {map_filename}')
    
    # Auto-open map in browser
    try:
        webbrowser.open(f'file://{os.path.abspath(map_filename)}')
        print("Map opened in browser")
    except:
        print("Could not auto-open browser, please manually open the map file")

if __name__ == '__main__':
    main()
