import os
import json
import math
import datetime
import logging
import csv
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def generateTimelineHTML(timePoints, sliceIndices, uniqueSlices, csvFilePath, viewName="Axial"):
    """Generate interactive timeline visualization for a single view.
    
    Args:
        timePoints: List of timestamps in format "YYYY-MM-DD HH:MM:SS.fff"
        sliceIndices: List of slice indices
        uniqueSlices: List of unique slice indices
        csvFilePath: Path to CSV file
        viewName: Name of the view (Axial, Sagittal, Coronal)
    
    Returns:
        str: HTML content with interactive visualization
    """
    logger.debug(f"Generating timeline for {viewName} view")
    logger.debug(f"Input data: {len(timePoints)} timePoints, {len(sliceIndices)} sliceIndices")
    logger.debug(f"First few timePoints: {timePoints[:3] if timePoints else 'None'}")
    
    if not timePoints:
        logger.error("No timePoints provided")
        return "<html><body><h1>No data found</h1></body></html>"
    
    # Convert timestamps to elapsed seconds
    try:
        start_time = datetime.datetime.strptime(timePoints[0], "%Y-%m-%d %H:%M:%S.%f")
        elapsed_times = []
        for timestamp in timePoints:
            current_time = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")
            elapsed = (current_time - start_time).total_seconds()
            elapsed_times.append(elapsed)
    except ValueError as e:
        logger.error(f"Error processing timestamps: {str(e)}")
        return f"<html><body><h1>Error processing timestamps: {str(e)}</h1></body></html>"
    
    # View-specific colors
    view_colors = {
        'Axial': '#cc0000',      # Red
        'Sagittal': '#0066cc',   # Blue
        'Coronal': '#009900'     # Green
    }
    color = view_colors.get(viewName, '#3b82f6')  # Default blue
    
    # Sort data by time
    sorted_indices = sorted(range(len(elapsed_times)), key=lambda i: elapsed_times[i])
    sorted_times = [elapsed_times[i] for i in sorted_indices]
    sorted_slices = [sliceIndices[i] for i in sorted_indices]
    
    # Convert time to minutes:seconds format for display
    formatted_times = []
    for t in sorted_times:
        minutes = int(t // 60)
        seconds = int(t % 60)
        formatted_times.append(f"{minutes}:{seconds:02d}")
    
    # Calculate summary statistics
    duration = max(sorted_times) - min(sorted_times) if sorted_times else 0
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    
    # Calculate y-axis range
    if uniqueSlices:
        slice_min = min(uniqueSlices)
        slice_max = max(uniqueSlices)
        # Ensure a minimum range even if all slices are the same
        if slice_min == slice_max:
            slice_min = max(0, slice_min - 5)
            slice_max = slice_max + 5
    else:
        slice_min, slice_max = 0, 10  # Default range if no data
    
    # Prepare data for D3
    data_points = []
    for i in range(len(sorted_times)):
        # Convert time to minutes:seconds format for display
        t = sorted_times[i]
        minutes_fmt = int(t // 60)
        seconds_fmt = int(t % 60)
        formatted_time = f"{minutes_fmt}:{seconds_fmt:02d}"
        
        data_points.append({
            "time": sorted_times[i],
            "slice": sorted_slices[i],
            "formatted_time": formatted_time
        })
    
    # Convert data to JSON for D3
    json_data = json.dumps(data_points)
    
    # Determine tick step for y-axis
    slice_range = slice_max - slice_min
    tick_step = max(1, math.ceil(slice_range / 8))  # Aim for ~8 ticks maximum
    
    # Create timestamp for the report
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create the HTML with embedded D3.js
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CT Slice Viewing Timeline - {viewName} View</title>
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f8f9fa;
            }}
            .container {{
                max-width: 900px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            .chart-container {{
                position: relative;
                margin-top: 20px;
            }}
            .title {{
                font-size: 20px;
                font-weight: bold;
                text-align: center;
                margin-bottom: 20px;
                color: #333;
            }}
            .axis-label {{
                font-size: 12px;
                font-weight: bold;
            }}
            .tooltip {{
                position: absolute;
                padding: 8px;
                background: rgba(0, 0, 0, 0.7);
                color: white;
                border-radius: 4px;
                font-size: 12px;
                pointer-events: none;
                opacity: 0;
                z-index: 1000;
            }}
            .info-section {{
                margin-top: 40px;
                padding: 20px;
                border: 1px solid #ddd;
                border-radius: 5px;
                background-color: #f9f9f9;
            }}
            .info-section h2 {{
                color: #333;
                margin-top: 0;
            }}
            .info-section h3 {{
                color: #555;
                margin-top: 20px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                padding: 8px;
                text-align: left;
                border: 1px solid #ddd;
            }}
            th {{
                background-color: #eee;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="title">CT Slice Viewing Timeline - {viewName} View</div>
            
            <div class="chart-container">
                <div id="chart"></div>
                <div id="tooltip" class="tooltip"></div>
            </div>
            
            <div class="info-section">
                <h2>CT Slice Navigation Analysis</h2>
                
                <div>
                    <h3>About This Report</h3>
                    <p>This report visualizes how a user navigated through CT slices during a viewing session. 
                       The graph shows slice positions over time, allowing for analysis of viewing patterns and focus areas.</p>
                    <p>Generated on: {current_time}</p>
                    <p>Data source: {os.path.basename(csvFilePath)}</p>
                </div>
                
                <div>
                    <h3>Interpretation Guide</h3>
                    <ul>
                        <li><strong>Horizontal plateaus</strong>: Time spent examining a particular slice</li>
                        <li><strong>Vertical jumps</strong>: Rapid navigation between distant slices</li>
                        <li><strong>Gradual slopes</strong>: Systematic examination of consecutive slices</li>
                        <li><strong>Recurring patterns</strong>: Areas revisited multiple times, potentially indicating regions of interest</li>
                    </ul>
                </div>
                
                <div>
                    <h3>Navigation Statistics</h3>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Total viewing time</td>
                            <td>{minutes} minutes, {seconds} seconds</td>
                        </tr>
                        <tr>
                            <td>Number of slices viewed</td>
                            <td>{len(uniqueSlices)}</td>
                        </tr>
                        <tr>
                            <td>Slice range</td>
                            <td>{min(uniqueSlices)} to {max(uniqueSlices)}</td>
                        </tr>
                        <tr>
                            <td>Data points recorded</td>
                            <td>{len(timePoints)}</td>
                        </tr>
                    </table>
                </div>
            </div>
        </div>

        <script>
            // Data from Python
            const data = {json_data};
            const viewColor = "{color}";
            const yMin = {slice_min - (slice_range * 0.05)};
            const yMax = {slice_max + (slice_range * 0.05)};
            const yTickStep = {tick_step};
            
            // Set up dimensions and margins
            const margin = {{top: 30, right: 30, bottom: 50, left: 60}};
            const width = 900 - margin.left - margin.right;
            const height = 500 - margin.top - margin.bottom;
            
            // Create SVG
            const svg = d3.select("#chart")
                .append("svg")
                    .attr("width", width + margin.left + margin.right)
                    .attr("height", height + margin.top + margin.bottom)
                .append("g")
                    .attr("transform", `translate(${{margin.left}},${{margin.top}})`);
            
            // Set up scales
            const x = d3.scaleLinear()
                .domain([0, d3.max(data, d => d.time)])
                .range([0, width]);
                
            const y = d3.scaleLinear()
                .domain([yMin, yMax])
                .range([height, 0]);
            
            // Add X axis with custom ticks (every ~10 seconds)
            const maxTime = d3.max(data, d => d.time);
            const xTickValues = [];
            const tickInterval = Math.max(10, Math.ceil(maxTime / 10)); // About 10 ticks total
            
            for (let i = 0; i <= maxTime; i += tickInterval) {{
                xTickValues.push(i);
            }}
            
            svg.append("g")
                .attr("transform", `translate(0,${{height}}`)
                .call(d3.axisBottom(x).tickValues(xTickValues).tickFormat(d => {{
                    const min = Math.floor(d / 60);
                    const sec = Math.floor(d % 60);
                    return `${{min}}:${{sec.toString().padStart(2, '0')}}`;
                }}));
            
            // Add X axis label
            svg.append("text")
                .attr("class", "axis-label")
                .attr("text-anchor", "middle")
                .attr("x", width / 2)
                .attr("y", height + margin.bottom - 10)
                .text("Time (minutes:seconds)");
            
            // Add Y axis with custom ticks
            const yTickValues = [];
            for (let i = Math.ceil(yMin); i <= Math.floor(yMax); i += yTickStep) {{
                yTickValues.push(i);
            }}
            
            svg.append("g")
                .call(d3.axisLeft(y).tickValues(yTickValues));
            
            // Add horizontal grid lines
            svg.selectAll("y-grid")
                .data(yTickValues)
                .enter()
                .append("line")
                    .attr("x1", 0)
                    .attr("x2", width)
                    .attr("y1", d => y(d))
                    .attr("y2", d => y(d))
                    .attr("stroke", "#e0e0e0")
                    .attr("stroke-width", 1);
            
            // Add vertical grid lines
            svg.selectAll("x-grid")
                .data(xTickValues)
                .enter()
                .append("line")
                    .attr("x1", d => x(d))
                    .attr("x2", d => x(d))
                    .attr("y1", 0)
                    .attr("y2", height)
                    .attr("stroke", "#e0e0e0")
                    .attr("stroke-width", 1);
                    
            // Create tooltip
            const tooltip = d3.select("#tooltip");
            
            // Add the line
            svg.append("path")
                .datum(data)
                .attr("fill", "none")
                .attr("stroke", viewColor)
                .attr("stroke-width", 2)
                .attr("d", d3.line()
                    .x(d => x(d.time))
                    .y(d => y(d.slice))
                );
                
                // Thêm vùng tương tác để hiển thị tooltip mà không cần các điểm
            svg.append("rect")
                .attr("width", width)
                .attr("height", height)
                    .attr("fill", "transparent")
                .style("pointer-events", "all")
                .on("mousemove", function(event) {{
                        const mousePos = d3.pointer(event);
                        const mouseX = mousePos[0];
                        const x0 = x.invert(mouseX);
                    
                        // Tìm điểm dữ liệu gần nhất
                        let closestPoint = null;
                        let minDistance = Infinity;
                    
                        for (let i = 0; i < data.length; i++) {{
                            const point = data[i];
                            const distance = Math.abs(point.time - x0);
                        if (distance < minDistance) {{
                            minDistance = distance;
                                closestPoint = point;
                        }}
                    }}
                    
                        if (closestPoint && minDistance < (d3.max(data, d => d[0]) / 50)) {{
                            tooltip.style("opacity", 1)
                                .html("Time: " + Math.floor(closestPoint[0] / 60) + ":" + Math.floor(closestPoint[0] % 60).toString().padStart(2, "0") + "<br>Slice: " + closestPoint[1])
                                .style("left", (event.pageX + 10) + "px")
                                .style("top", (event.pageY - 28) + "px");
                    }} else {{
                        tooltip.style("opacity", 0);
                    }}
                }})
                    .on("mouseout", function() {{
                    tooltip.style("opacity", 0);
                }});
                
            // Add horizontal scrolling with mouse wheel
            document.querySelector("#chart svg").addEventListener("wheel", function(event) {{
                event.preventDefault();
                
                // Calculate current domain of x-axis
                const currentDomain = x.domain();
                const domainWidth = currentDomain[1] - currentDomain[0];
                
                // Calculate zoom factor based on wheel delta
                const zoomFactor = event.deltaY > 0 ? 1.1 : 0.9;
                
                // Calculate how much we need to move the domain
                const moveAmount = (domainWidth * zoomFactor - domainWidth) / 2;
                
                // Only allow zooming if we're still seeing most of the data
                if ((zoomFactor > 1 && domainWidth > (maxTime / 5)) || // Zoom out limit
                    (zoomFactor < 1 && domainWidth * zoomFactor > (maxTime / 50))) {{ // Zoom in limit
                    
                    // Update the x-axis domain
                    x.domain([
                        Math.max(0, currentDomain[0] - moveAmount),
                        Math.min(maxTime, currentDomain[1] + moveAmount)
                    ]);
                    
                    // Update all elements that depend on the x-axis
                    svg.select("g").call(d3.axisBottom(x).tickValues(xTickValues).tickFormat(d => {{
                        const min = Math.floor(d / 60);
                        const sec = Math.floor(d % 60);
                        return `${{min}}:${{sec.toString().padStart(2, '0')}}`;
                    }}));
                    
                    // Update grid lines
                    svg.selectAll("line[y1='0']").each(function(d, i) {{
                        d3.select(this).attr("x1", x(xTickValues[i])).attr("x2", x(xTickValues[i]));
                    }});
                    
                    // Update the line
                    svg.select("path").attr("d", d3.line()
                        .x(d => x(d.time))
                        .y(d => y(d.slice))
                    );
                }}
            }}, {{ passive: false }});
        </script>
    </body>
    </html>
    """
    
    return html

def generateCombinedTimelineHTML(viewsData, transcriptionData=None, classificationData=None):
    """Generate HTML content for combined timeline visualization.
    
    Args:
        viewsData (dict): Dictionary containing data for each view
        transcriptionData (list, optional): List of transcription segments
        classificationData (list, optional): List of classification data from curve.csv
    
    Returns:
        str: HTML content as string
    """
    import json
    import os
    import pandas as pd
    
    if not viewsData:
        print("[ERROR] No view data provided")
        return None
        
    print(f"[DEBUG] Received viewsData: {viewsData}")
    
    # Get global time range
    all_times = []
    for view_data in viewsData.values():
        all_times.extend(view_data['times'])
    time_min = min(all_times) if all_times else 0
    time_max = max(all_times) if all_times else 1
    
    print(f"[DEBUG] Time range: {time_min} to {time_max}")
    
    # Prepare view colors
    view_colors = {
        'Axial': '#ff0000',     # Red
        'Sagittal': '#0000ff',  # Blue 
        'Coronal': '#00ff00'    # Green
    }
    
    # Prepare classification colors
    class_colors = {
        1: '#ffcccb',  # Light red
        2: '#ffebcc',  # Light orange
        3: '#ddffcc',  # Light green
        4: '#cce5ff',  # Light blue
        5: '#e5ccff'   # Light purple
    }
    
    # Check if classification file exists and try to load it
    if classificationData is None:
        print("[DEBUG] No classification data provided, looking for classification file")
        try:
            # Get directory of first view data
            for view_name, view_data in viewsData.items():
                if 'csv_path' in view_data:
                    session_dir = os.path.dirname(view_data['csv_path'])
                    if session_dir:
                        print(f"[DEBUG] Looking for classification files in {session_dir}")
                        # Try to find classification file
                        for file in os.listdir(session_dir):
                            if file.endswith('_classification.csv'):
                                classification_path = os.path.join(session_dir, file)
                                try:
                                    print(f"[DEBUG] Trying to load classification data from {classification_path}")
                                    classification_df = pd.read_csv(classification_path)
                                    classificationData = classification_df.to_dict('records')
                                    print(f"[INFO] Loaded classification data from {classification_path}")
                                    print(f"[DEBUG] Sample classification data: {classificationData[:2]}")
                                    break
                                except Exception as e:
                                    print(f"[ERROR] Failed to load classification data: {e}")
                        break
        except Exception as e:
            print(f"[WARNING] Could not load classification data: {e}")
    else:
        print(f"[DEBUG] Classification data provided directly: {len(classificationData)} records")
        print(f"[DEBUG] Sample classification data: {classificationData[:2] if classificationData else []}")
    
    # Convert Python data to JSON strings for JavaScript
    viewsData_json = json.dumps(viewsData)
    transcriptionData_json = json.dumps(transcriptionData if transcriptionData else [])
    # transcriptionData_json = json.dumps(transcriptionData) if transcriptionData else [] )
    print("--------------------------------")
    print(f"[DEBUG] ON PLOT GENERATOR: {transcriptionData_json}")
    view_colors_json = json.dumps(view_colors)
    class_colors_json = json.dumps(class_colors)
    classificationData_json = json.dumps(classificationData if classificationData else [])
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>CT Slice Viewing Timeline</title>
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f8f9fa;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            .title {{
                font-size: 24px;
                font-weight: bold;
                text-align: center;
                margin-bottom: 20px;
                color: #333;
            }}
            .view-container {{
                margin-bottom: 20px;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
            }}
            .class-container {{
                margin-bottom: 20px;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
            }}
            .view-title {{
                font-size: 16px;
                font-weight: bold;
                margin-bottom: 10px;
                color: #555;
            }}
            .grid line {{
                stroke: #ddd;
                stroke-opacity: 0.7;
            }}
            .transcription-container {{
                margin-top: 20px;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
            }}
            .transcription-box {{
                fill: #ffffcc;
                opacity: 0.8;
                stroke: #000000;
                stroke-width: 1;
                cursor: pointer;
            }}
            .transcription-box:hover {{
                opacity: 1;
                fill: #ffff99;
            }}
            .transcription-text {{
                font-size: 11px;
                fill: #000;
                pointer-events: none;
                dominant-baseline: middle;
                text-anchor: start;
                font-weight: 500;
                text-shadow: 0 0 2px white, 0 0 2px white, 0 0 2px white, 0 0 2px white;
                overflow: hidden;
                text-overflow: ellipsis;
            }}
            .text-container {{
                overflow: hidden;
                word-wrap: break-word;
                white-space: normal;
                font-family: Arial, sans-serif;
                font-size: 11px;
                padding: 3px;
                color: #000;
                text-shadow: 0 0 2px white, 0 0 2px white, 0 0 2px white, 0 0 2px white;
            }}
            .tooltip {{
                position: absolute;
                padding: 8px;
                background: rgba(0, 0, 0, 0.7);
                color: white;
                border-radius: 4px;
                font-size: 12px;
                pointer-events: none;
                opacity: 0;
                max-width: 300px;
                white-space: normal;
                word-wrap: break-word;
                z-index: 1000;
            }}
            .controls {{
                text-align: center;
                margin-top: 20px;
            }}
            button {{
                padding: 8px 16px;
                font-size: 14px;
                cursor: pointer;
            }}
            .class-segment {{
                stroke: #000;
                stroke-width: 1;
                opacity: 0.7;
            }}
            .class-segment:hover {{
                opacity: 1;
                stroke-width: 2;
            }}
            .class-text {{
                font-size: 14px;
                font-weight: bold;
                text-anchor: middle;
                dominant-baseline: middle;
            }}
            .class-legend {{
                margin: 10px 0;
                display: flex;
                justify-content: center;
                flex-wrap: wrap;
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                margin: 0 10px;
            }}
            .legend-color {{
                width: 20px;
                height: 20px;
                margin-right: 5px;
                border: 1px solid #000;
                border-radius: 3px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="title">CT Slice Viewing Timeline</h1>
            
            <div id="views-container">
                <!-- Views will be inserted here -->
            </div>
            
            <div id="class-container" class="class-container">
                <div class="view-title">Curve classification</div>
                <div class="class-legend" id="class-legend"></div>
                <!-- Classification will be inserted here -->
            </div>
            
            <div id="transcription-container" class="transcription-container">
                <div class="view-title">Spoken Text</div>
                <!-- Transcription will be inserted here -->
            </div>
        </div>
        
        <script>
        // Data
        const viewsData = {viewsData_json};
        const transcriptionData = {transcriptionData_json};
        const classificationData = {classificationData_json};
        const viewColors = {view_colors_json};
        const classColors = {class_colors_json};
        
        // Constants
        const MARGIN = {{top: 20, right: 30, bottom: 30, left: 50}};
        const WIDTH = 1100 - MARGIN.left - MARGIN.right;
        const VIEW_HEIGHT = 150;
        const CLASS_HEIGHT = 40;
        const TRANS_HEIGHT = 80;
        
        // Create time scale (shared between all views)
        const timeScale = d3.scaleLinear()
            .domain([{time_min}, {time_max}])
            .range([0, WIDTH]);
            
        // Create container
        const container = d3.select("#views-container");
        
        // Create views
        const viewOrder = ['Sagittal', 'Coronal','Axial'];
        viewOrder.forEach(viewName => {{
            const data = viewsData[viewName];
            if (!data) return;
            
            // Create view container
            const viewContainer = container.append("div")
                .attr("class", "view-container")
                .attr("id", `${{viewName.toLowerCase()}}-view`);
                
            viewContainer.append("div")
                .attr("class", "view-title")
                .text(`${{viewName}} View`);
                
            // Create SVG
            const svg = viewContainer.append("svg")
                .attr("width", WIDTH + MARGIN.left + MARGIN.right)
                .attr("height", VIEW_HEIGHT + MARGIN.top + MARGIN.bottom)
                .append("g")
                .attr("transform", `translate(${{MARGIN.left}},${{MARGIN.top}})`);
                
            // Create scales
            const yScale = d3.scaleLinear()
                .domain([d3.min(data.indices), d3.max(data.indices)])
                .range([VIEW_HEIGHT, 0]);
                
            // Add grid
            svg.append("g")
                .attr("class", "grid")
                .attr("transform", `translate(0,${{VIEW_HEIGHT}})`)
                .call(d3.axisBottom(timeScale)
                    .ticks(10)
                    .tickSize(-VIEW_HEIGHT)
                    .tickFormat(""));
                    
            // Create line generator
            const line = d3.line()
                .x(d => timeScale(d[0]))
                .y(d => yScale(d[1]))
                .curve(d3.curveMonotoneX);
                
            // Add line path
            const points = data.times.map((t, i) => [t, data.indices[i]]);
            
            svg.append("path")
                .datum(points)
                .attr("fill", "none")
                .attr("stroke", viewColors[viewName])
                .attr("stroke-width", 2)
                .attr("d", line);
                
            // Add axes
            svg.append("g")
                .attr("transform", `translate(0,${{VIEW_HEIGHT}})`)
                .call(d3.axisBottom(timeScale)
                    .tickFormat(d => {{
                        const minutes = Math.floor(d / 60);
                        const seconds = Math.floor(d % 60);
                        return `${{minutes}}:${{seconds.toString().padStart(2, '0')}}`;
                    }}));
                    
            svg.append("g")
                .call(d3.axisLeft(yScale));
                
            // Add tooltips for points
            const tooltip = d3.select("body").append("div")
                .attr("class", "tooltip");
                
            // Thêm vùng tương tác để hiển thị tooltip mà không cần các điểm
            svg.append("rect")
                .attr("width", WIDTH)
                .attr("height", VIEW_HEIGHT)
                .attr("fill", "transparent")
                .style("pointer-events", "all")
                .on("mousemove", function(event) {{
                    const mousePos = d3.pointer(event);
                    const mouseX = mousePos[0];
                    const x0 = timeScale.invert(mouseX);
                    
                    // Tìm điểm dữ liệu gần nhất
                    let closestPoint = null;
                    let minDistance = Infinity;
                    
                    for (let i = 0; i < points.length; i++) {{
                        const point = points[i];
                        const distance = Math.abs(point[0] - x0);
                        if (distance < minDistance) {{
                            minDistance = distance;
                            closestPoint = point;
                        }}
                    }}
                    
                    if (closestPoint && minDistance < (d3.max(data, d => d[0]) / 50)) {{
                        tooltip.style("opacity", 1)
                            .html("Time: " + Math.floor(closestPoint[0] / 60) + ":" + Math.floor(closestPoint[0] % 60).toString().padStart(2, "0") + "<br>Slice: " + closestPoint[1])
                            .style("left", (event.pageX + 10) + "px")
                            .style("top", (event.pageY - 28) + "px");
                    }} else {{
                        tooltip.style("opacity", 0);
                    }}
                }})
                .on("mouseout", function() {{
                    tooltip.style("opacity", 0);
                }});
        }});
        
        // Create classification visualization
        if (classificationData && classificationData.length > 0) {{
            // Create class container
            const classContainer = d3.select("#class-container");
            
            // Create legend for classes
            const legend = d3.select("#class-legend");
            
            // Add legend items
            for (let i = 1; i <= 5; i++) {{
                const legendItem = legend.append("div")
                    .attr("class", "legend-item");
                    
                legendItem.append("div")
                    .attr("class", "legend-color")
                    .style("background-color", classColors[i]);
                    
                legendItem.append("div")
                    .text(`Class ${{i}}`);
            }}
            
            // Create SVG
            const svg = classContainer.append("svg")
                .attr("width", WIDTH + MARGIN.left + MARGIN.right)
                .attr("height", CLASS_HEIGHT + MARGIN.top + MARGIN.bottom)
                .append("g")
                .attr("transform", `translate(${{MARGIN.left}},${{MARGIN.top}})`);
                
            // Add grid
            svg.append("g")
                .attr("class", "grid")
                .attr("transform", `translate(0,${{CLASS_HEIGHT}})`)
                .call(d3.axisBottom(timeScale)
                    .ticks(10)
                    .tickSize(-CLASS_HEIGHT)
                    .tickFormat(""));
                    
            // Add class segments
            svg.selectAll(".class-segment")
                .data(classificationData)
                .enter()
                .append("rect")
                .attr("class", "class-segment")
                .attr("x", d => timeScale(parseFloat(d.start_time)))
                .attr("y", 10)
                .attr("width", d => Math.max(2, timeScale(parseFloat(d.end_time)) - timeScale(parseFloat(d.start_time))))
                .attr("height", CLASS_HEIGHT - 20)
                .attr("fill", d => classColors[parseInt(d.class)] || "#cccccc")
                .attr("rx", 5)
                .attr("ry", 5)
                .on("mouseover", function(event, d) {{
                    tooltip.transition()
                        .duration(200)
                        .style("opacity", .9);
                    tooltip.html(`Class: ${{d.class}}<br>Start: ${{parseFloat(d.start_time).toFixed(2)}}s<br>End: ${{parseFloat(d.end_time).toFixed(2)}}s`)
                        .style("left", (event.pageX + 10) + "px")
                        .style("top", (event.pageY - 28) + "px");
                }})
                .on("mouseout", function() {{
                    tooltip.transition()
                        .duration(500)
                        .style("opacity", 0);
        }});
        
            // Add class labels (numbers)
            svg.selectAll(".class-text")
                .data(classificationData)
                .enter()
                .append("text")
                .attr("class", "class-text")
                .attr("x", d => timeScale(parseFloat(d.start_time)) + (timeScale(parseFloat(d.end_time)) - timeScale(parseFloat(d.start_time))) / 2)
                .attr("y", CLASS_HEIGHT / 2)
                .text(d => d.class)
                .attr("fill", "#000")
                .style("pointer-events", "none");
                
            // Add axes
            svg.append("g")
                .attr("transform", `translate(0,${{CLASS_HEIGHT}})`)
                .call(d3.axisBottom(timeScale)
                    .tickFormat(d => {{
                        const minutes = Math.floor(d / 60);
                        const seconds = Math.floor(d % 60);
                        return `${{minutes}}:${{seconds.toString().padStart(2, '0')}}`;
                    }}));
        }} else {{
            // No classification data
            d3.select("#class-container")
                .append("div")
                .style("padding", "20px")
                .style("text-align", "center")
                .text("Not data classification");
        }}
        
        // Create transcription visualization
        if (transcriptionData && transcriptionData.length > 0) {{
            const svg = d3.select("#transcription-container")
                .append("svg")
                .attr("width", WIDTH + MARGIN.left + MARGIN.right)
                .attr("height", TRANS_HEIGHT)
                .append("g")
                .attr("transform", `translate(${{MARGIN.left}},0)`);
                
            // Add transcription boxes
            svg.selectAll("rect")
                .data(transcriptionData)
                .enter()
                .append("rect")
                .attr("class", "transcription-box")
                .attr("x", d => timeScale(d.start_time))
                .attr("y", 0)
                .attr("width", d => timeScale(d.end_time) - timeScale(d.start_time))
                .attr("height", TRANS_HEIGHT - 20)
                .on("mouseover", function(event, d) {{
                    const minutes = Math.floor(d.start_time / 60);
                    const seconds = Math.floor(d.start_time % 60);
                    const timeStr = `${{minutes.toString().padStart(2, '0')}}:${{seconds.toString().padStart(2, '0')}}`;
                    
                    d3.select("body").append("div")
                        .attr("class", "tooltip")
                        .style("opacity", 1)
                        .style("left", (event.pageX + 10) + "px")
                        .style("top", (event.pageY - 28) + "px")
                        .html(`Time: ${{timeStr}}<br>${{d.text}}`);
                }})
                .on("mouseout", function() {{
                    d3.selectAll(".tooltip").remove();
                }});
                
            // Add text labels inside the boxes with word wrapping
            svg.selectAll("foreignObject")
                .data(transcriptionData)
                .enter()
                .append("foreignObject")
                .attr("x", d => timeScale(d.start_time) + 1)
                .attr("y", 1)
                .attr("width", d => Math.max(30, timeScale(d.end_time) - timeScale(d.start_time) - 2))
                .attr("height", TRANS_HEIGHT - 22)
                .append("xhtml:div")
                .attr("class", "text-container")
                .style("height", (TRANS_HEIGHT - 22) + "px")
                .html(d => d.text);
            // Add time axis
            svg.append("g")
                .attr("transform", `translate(0,${{TRANS_HEIGHT - 20}})`)
                .call(d3.axisBottom(timeScale)
                    .tickFormat(d => {{
                        const minutes = Math.floor(d / 60);
                        const seconds = Math.floor(d % 60);
                        return `${{minutes.toString().padStart(2, '0')}}:${{seconds.toString().padStart(2, '0')}}`;
                    }}));
        }}
        </script>
    </body>
    </html>
    """
    
    print("[DEBUG] Generated HTML content")
    return html_content

def generateTimelineHTML(csvFilePath, viewName="Axial"):
    """Generate interactive timeline visualization for a single view.
    
    Args:
        csvFilePath: Path to CSV file
        viewName: Name of the view (Axial, Sagittal, Coronal)
    
    Returns:
        str: HTML content with interactive visualization
    """
    print(f"generateTimelineHTML called with csvFilePath={csvFilePath}, viewName={viewName}")
    
    try:
        # Read CSV file
        print("Reading CSV file...")
        timePoints = []
        sliceIndices = []
        
        with open(csvFilePath, 'r') as csvFile:
            reader = csv.DictReader(csvFile)
            for row in reader:
                try:
                    timePoints.append(row['timestamp'])
                    sliceIndices.append(int(row['slice_number']))
                except (KeyError, ValueError) as e:
                    print(f"Error reading row: {e}")
                    print(f"Row data: {row}")
        
        print(f"Read {len(timePoints)} data points")
        uniqueSlices = sorted(set(sliceIndices))
        print(f"Found {len(uniqueSlices)} unique slices")
        
        # Generate the HTML content
        logger.debug(f"Generating timeline for {viewName} view")
        logger.debug(f"Input data: {len(timePoints)} timePoints, {len(sliceIndices)} sliceIndices")
        logger.debug(f"First few timePoints: {timePoints[:3] if timePoints else 'None'}")
        
        if not timePoints:
            logger.error("No timePoints provided")
            return "<html><body><h1>No data found</h1></body></html>"
        
        # Convert timestamps to elapsed seconds
        try:
            start_time = datetime.datetime.strptime(timePoints[0], "%Y-%m-%d %H:%M:%S.%f")
            elapsed_times = []
            for timestamp in timePoints:
                current_time = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")
                elapsed = (current_time - start_time).total_seconds()
                elapsed_times.append(elapsed)
        except ValueError as e:
            logger.error(f"Error processing timestamps: {str(e)}")
            return f"<html><body><h1>Error processing timestamps: {str(e)}</h1></body></html>"
        
        # View-specific colors
        view_colors = {
            'Axial': '#cc0000',      # Red
            'Sagittal': '#0066cc',   # Blue
            'Coronal': '#009900'     # Green
        }
        color = view_colors.get(viewName, '#3b82f6')  # Default blue
        
        # Sort data by time
        sorted_indices = sorted(range(len(elapsed_times)), key=lambda i: elapsed_times[i])
        sorted_times = [elapsed_times[i] for i in sorted_indices]
        sorted_slices = [sliceIndices[i] for i in sorted_indices]
        
        # Convert time to minutes:seconds format for display
        formatted_times = []
        for t in sorted_times:
            minutes = int(t // 60)
            seconds = int(t % 60)
            formatted_times.append(f"{minutes}:{seconds:02d}")
        
        # Calculate summary statistics
        duration = max(sorted_times) - min(sorted_times) if sorted_times else 0
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        
        # Calculate y-axis range
        if uniqueSlices:
            slice_min = min(uniqueSlices)
            slice_max = max(uniqueSlices)
            # Ensure a minimum range even if all slices are the same
            if slice_min == slice_max:
                slice_min = max(0, slice_min - 5)
                slice_max = slice_max + 5
        else:
            slice_min, slice_max = 0, 10  # Default range if no data
        
        # Prepare data for D3
        data_points = []
        for i in range(len(sorted_times)):
            # Convert time to minutes:seconds format for display
            t = sorted_times[i]
            minutes_fmt = int(t // 60)
            seconds_fmt = int(t % 60)
            formatted_time = f"{minutes_fmt}:{seconds_fmt:02d}"
            
            data_points.append({
                "time": sorted_times[i],
                "slice": sorted_slices[i],
                "formatted_time": formatted_time
            })
        
        # Convert data to JSON for D3
        json_data = json.dumps(data_points)
        
        # Determine tick step for y-axis
        slice_range = slice_max - slice_min
        tick_step = max(1, math.ceil(slice_range / 8))  # Aim for ~8 ticks maximum
        
        # Create timestamp for the report
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create the HTML content
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>CT Slice Viewing Timeline - {viewName} View</title>
            <script src="https://d3js.org/d3.v7.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f8f9fa;
                }}
                .container {{
                    max-width: 900px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 5px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                }}
                .chart-container {{
                    position: relative;
                    margin-top: 20px;
                }}
                .title {{
                    font-size: 20px;
                    font-weight: bold;
                    text-align: center;
                    margin-bottom: 20px;
                    color: #333;
                }}
                .axis-label {{
                    font-size: 12px;
                    font-weight: bold;
                }}
                .tooltip {{
                    position: absolute;
                    padding: 8px;
                    background: rgba(0, 0, 0, 0.7);
                    color: white;
                    border-radius: 4px;
                    font-size: 12px;
                    pointer-events: none;
                    opacity: 0;
                    z-index: 1000;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="title">CT Slice Viewing Timeline - {viewName} View</div>
                <div class="chart-container">
                    <div id="chart"></div>
                    <div id="tooltip" class="tooltip"></div>
                </div>
            </div>

            <script>
                /* Show tooltip with transcription text */
                const data = {json_data};
                const viewColor = "{color}";
                const yMin = {slice_min - (slice_range * 0.05)};
                const yMax = {slice_max + (slice_range * 0.05)};
                const yTickStep = {tick_step};
                
                // Set up dimensions and margins
                const margin = {{top: 30, right: 30, bottom: 50, left: 60}};
                const width = 900 - margin.left - margin.right;
                const height = 500 - margin.top - margin.bottom;
                
                // Create SVG
                const svg = d3.select("#chart")
                    .append("svg")
                        .attr("width", width + margin.left + margin.right)
                        .attr("height", height + margin.top + margin.bottom)
                    .append("g")
                        .attr("transform", `translate(${{margin.left}},${{margin.top}})`);
                
                // Set up scales
                const x = d3.scaleLinear()
                    .domain([0, d3.max(data, d => d.time)])
                    .range([0, width]);
                    
                const y = d3.scaleLinear()
                    .domain([yMin, yMax])
                    .range([height, 0]);
                
                // Add X axis
                svg.append("g")
                    .attr("transform", `translate(0,${{height}})`)
                    .call(d3.axisBottom(x).tickFormat(t => {{
                        const min = Math.floor(t / 60);
                        const sec = Math.floor(t % 60);
                        return `${{min}}:${{sec.toString().padStart(2, '0')}}`;
                    }}));
                
                // Add Y axis
                svg.append("g")
                    .call(d3.axisLeft(y));
                
                // Add the line
                svg.append("path")
                    .datum(data)
                    .attr("fill", "none")
                    .attr("stroke", viewColor)
                    .attr("stroke-width", 2)
                    .attr("d", d3.line()
                        .x(d => x(d.time))
                        .y(d => y(d.slice))
                    );
                
                // Add tooltip functionality
                const tooltip = d3.select("#tooltip");
                
                // Thêm vùng tương tác để hiển thị tooltip mà không cần các điểm
                svg.append("rect")
                    .attr("width", width)
                    .attr("height", height)
                    .attr("fill", "transparent")
                    .style("pointer-events", "all")
                    .on("mousemove", function(event) {{
                        const mousePos = d3.pointer(event);
                        const mouseX = mousePos[0];
                        const x0 = x.invert(mouseX);
                        
                        // Tìm điểm dữ liệu gần nhất
                        let closestPoint = null;
                        let minDistance = Infinity;
                        
                        for (let i = 0; i < data.length; i++) {{
                            const point = data[i];
                            const distance = Math.abs(point.time - x0);
                            if (distance < minDistance) {{
                                minDistance = distance;
                                closestPoint = point;
                            }}
                        }}
                        
                        if (closestPoint && minDistance < (d3.max(data, d => d[0]) / 50)) {{
                        tooltip.style("opacity", 1)
                                .html("Time: " + Math.floor(closestPoint[0] / 60) + ":" + Math.floor(closestPoint[0] % 60).toString().padStart(2, "0") + "<br>Slice: " + closestPoint[1])
                            .style("left", (event.pageX + 10) + "px")
                            .style("top", (event.pageY - 28) + "px");
                        }} else {{
                            tooltip.style("opacity", 0);
                        }}
                    }})
                    .on("mouseout", function() {{
                        tooltip.style("opacity", 0);
                    }});
            </script>
        </body>
        </html>
        """
        
        return html
        
    except Exception as e:
        print(f"Error in generateTimelineHTML: {str(e)}")
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())
        return None

def get_transcription_segments(transcriptionData):
    if isinstance(transcriptionData, dict) and 'segments' in transcriptionData:
        return transcriptionData['segments']
    elif isinstance(transcriptionData, list):
        return transcriptionData
    else:
        return []






