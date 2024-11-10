import streamlit as st
import pandas as pd
from openai import OpenAI
import os
import json
import logging
import traceback
from typing import Optional, Dict, List, Type, Any
import re
import urllib.parse
from streamlit import components
from abc import ABC, abstractmethod
from datetime import datetime
from threading import Lock
import asyncio
from typing import Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define MAX_WORKFLOW_HISTORY constant
MAX_WORKFLOW_HISTORY = 20

# Initialize session state
if 'workflow_history' not in st.session_state:
    st.session_state.workflow_history = []  # Stores the history of visualization changes
if 'current_viz' not in st.session_state:
    st.session_state.current_viz = None  # Stores the current D3.js visualization code
if 'preprocessed_df' not in st.session_state:
    st.session_state.preprocessed_df = None  # Stores the preprocessed DataFrame
if 'update_viz' not in st.session_state:
    st.session_state.update_viz = False  # Flag to trigger visualization update
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []  # Stores the chat history

def display_loading_animation():
    loading_html = """
    <div class="loading-container" style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 500px;">
        <div class="loading-spinner">
            <div class="spinner-ring"></div>
            <div class="spinner-ring"></div>
            <div class="spinner-ring"></div>
            <div class="spinner-ring"></div>
        </div>
        <div class="loading-text">LLM gods are doing magic now...</div>
    </div>
    <style>
        .loading-spinner {
            position: relative;
            width: 80px;
            height: 80px;
        }
        .spinner-ring {
            position: absolute;
            width: 100%;
            height: 100%;
            border: 4px solid transparent;
            border-top-color: #3498db;
            border-radius: 50%;
            animation: spin 1.2s cubic-bezier(0.5, 0, 0.5, 1) infinite;
        }
        .spinner-ring:nth-child(1) { animation-delay: -0.45s; }
        .spinner-ring:nth-child(2) { animation-delay: -0.3s; }
        .spinner-ring:nth-child(3) { animation-delay: -0.15s; }
        .loading-text {
            margin-top: 20px;
            font-family: Arial, sans-serif;
            font-size: 18px;
            color: #3498db;
            animation: pulse 1.5s ease-in-out infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        @keyframes pulse {
            0%, 100% { opacity: 0.5; }
            50% { opacity: 1; }
        }
    </style>
    """
    return st.components.v1.html(loading_html, height=400)

def get_api_credentials() -> Dict[str, str]:
    """
    Retrieve API keys for OpenAI, Google, and Anthropic from Streamlit secrets
    or prompt the user to enter them via sidebar inputs.

    Returns:
        Dict[str, str]: A dictionary containing API keys for each provider.
    """
    api_keys = {
        "openai": st.secrets.get("OPENAI_API_KEY") or st.sidebar.text_input(
            "Enter your OpenAI API Key", type="password"
        ),
        "google": st.secrets.get("GOOGLE_API_KEY") or st.sidebar.text_input(
            "Enter your Google API Key", type="password"
        ),
        "anthropic": st.secrets.get("ANTHROPIC_API_KEY") or st.sidebar.text_input(
            "Enter your Anthropic API Key", type="password"
        ),
    }

    # Display success messages if keys are provided
    for provider, key in api_keys.items():
        if key:
            st.sidebar.success(f"{provider.capitalize()} API key received successfully! 🎉")

    return api_keys

def test_api_key(api_key: str) -> bool:
    """
    Test if the provided API key is valid.
    
    This function attempts to list OpenAI models using the provided API key.
    If successful, the key is considered valid.
    
    Args:
        api_key (str): The OpenAI API key to test.
    
    Returns:
        bool: True if the API key is valid, False otherwise.
    """
    client = OpenAI(api_key=api_key)
    try:
        client.models.list()
        return True
    except Exception as e:
        logger.error(f"API key validation failed: {str(e)}")
        return False

def preprocess_data(file1, file2) -> pd.DataFrame:
    """
    Preprocess and merge the two dataframes for comparison.
    
    This function reads two CSV files, adds a 'Source' column to each,
    merges them, handles missing values, ensures consistent data types,
    and standardizes column names.
    
    Args:
        file1: First CSV file uploaded by the user.
        file2: Second CSV file uploaded by the user.
    
    Returns:
        pd.DataFrame: Preprocessed and merged DataFrame.
    
    Raises:
        ValueError: If files are empty or cannot be parsed.
        Exception: For any other preprocessing errors.
    """
    logger.info("Starting data preprocessing")
    try:
        # Read CSV files into pandas DataFrames
        try:
            df1 = pd.read_csv(file1)
            df2 = pd.read_csv(file2)
        except pd.errors.EmptyDataError:
            raise ValueError("One or both of the uploaded files are empty.")
        except pd.errors.ParserError:
            raise ValueError("Error parsing the CSV files. Please ensure they are valid CSV format.")
        
        # Add 'Source' column to identify the origin of each row
        df1['Source'] = 'CSV file 1'
        df2['Source'] = 'CSV file 2'
        
        # Merge the two DataFrames
        merged_df = pd.concat([df1, df2], ignore_index=True)
        
        # Handle missing values by filling them with 0
        merged_df = merged_df.fillna(0)
        
        # Ensure consistent data types
        for col in merged_df.columns:
            if merged_df[col].dtype == 'object':
                try:
                    merged_df[col] = pd.to_numeric(merged_df[col])
                except ValueError:
                    pass  # Keep as string if can't convert to numeric
        
        # Standardize column names: lowercase and replace spaces with underscores
        merged_df.columns = merged_df.columns.str.lower().str.replace(' ', '_')
        
        logger.info("Data preprocessing completed successfully")
        return merged_df
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        raise

def validate_d3_code(code: str) -> bool:
    """
    Perform basic validation on the generated D3 code.
    
    This function checks for the presence of key D3.js elements and syntax.
    
    Args:
        code (str): The D3.js code to validate.
    
    Returns:
        bool: True if the code passes basic validation, False otherwise.
    """
    # Check if the code defines the createVisualization function
    if not re.search(r'function\s+createVisualization\s*\(data,\s*svgElement\)\s*{', code):
        return False
    
    # Check for basic D3 v7 method calls
    d3_methods = ['d3.select', 'd3.scaleLinear', 'd3.axisBottom', 'd3.axisLeft']
    if not any(method in code for method in d3_methods):
        return False
    
    # Check for balanced braces
    if code.count('{') != code.count('}'):
        return False
    
    return True

def generate_d3_code_openai(
    df: pd.DataFrame,
    api_key: str,
    user_input: str = "",
    viz_type: str = "Bar Chart"
) -> str:
    """
    Generate D3.js code using OpenAI's API.

    Args:
        df (pd.DataFrame): The preprocessed DataFrame.
        api_key (str): OpenAI API key.
        user_input (str, optional): Additional user requirements for visualization.
        viz_type (str, optional): Type of visualization to generate.

    Returns:
        str: Generated D3.js code.
    """
    logger.info("Starting D3 code generation with OpenAI")
    data_sample = df.head(5).to_dict(orient='records')
    schema = df.dtypes.to_dict()
    schema_str = "\n".join([f"{col}: {dtype}" for col, dtype in schema.items()])

    client = OpenAI(api_key=api_key)  # Assuming OpenAI SDK is used

    base_prompt = f"""
    # D3.js Code Generation Task

    Generate ONLY D3.js version 7 code for a {viz_type.lower()}, interactive, and comparative visualization. Do not include any explanations, comments, or markdown formatting.

    Critical Requirements:
    1. Create a function named createVisualization(data, svgElement)
    2. Implement a responsive SVG that adjusts its size based on the content
    3. Utilize the full width and height provided (1000x600 pixels)
    4. Implement zoom, pan, and brush functionality for exploring the data
    5. Ensure efficient use of space, minimizing empty areas
    6. Add appropriate margins, title, axes labels, and an interactive legend
    7. Use different colors for each data source and implement a color scale
    8. Implement tooltips showing full information on hover
    9. Include smooth transitions for any data updates or interactions
    10. Handle potential overlapping of data points or labels
    11. Always have grid lines on the visualization
    12. Animate the visualization as much as possible
    13. Utilize d3.select() for DOM manipulation and d3.data() for data binding
    14. Implement advanced scales: d3.scaleLinear(), d3.scaleBand(), d3.scaleTime(), d3.scaleOrdinal(d3.schemeCategory10)
    15. Create dynamic, animated axes using d3.axisBottom(), d3.axisLeft() with custom tick formatting
    16. Implement smooth transitions and animations using d3.transition() and d3.easeCubic
    17. Utilize d3.line(), d3.area(), d3.arc() for creating complex shapes and paths
    18. Implement interactivity: d3.brush(), d3.zoom(), d3.drag() for user interaction
    19. Use d3.interpolate() for smooth color and value transitions
    20. Implement d3.forceSimulation() for force-directed graph layouts if applicable
    21. Utilize d3.geoPath() and d3.geoProjection() for geographical visualizations if applicable
    22. Use d3.contours() and d3.density2D() for density and contour visualizations if applicable
    23. Implement d3.voronoi() for proximity-based visualizations if applicable
    24. Utilize d3.chord() and d3.ribbon() for relationship visualizations if applicable
    25. Implement advanced event handling with d3.on() for mouseover, click, etc.
    26. Use d3.format() for number formatting in tooltips and labels
    27. Implement d3.timeFormat() for date/time formatting if applicable
    28. Utilize d3.range() and d3.shuffle() for data generation and randomization if needed
    29. Implement d3.nest() for data restructuring and aggregation if needed
    30. Use d3.queue() for asynchronous data loading and processing
    31. Implement accessibility features using ARIA attributes
    32. Optimize performance using d3.quadtree() for spatial indexing if applicable
    33. Implement responsive design using d3.select(window).on("resize", ...)
    34. Focus on creating a comparative visualization that highlights data differences
    35. Implement error handling for invalid data formats and gracefully handle missing data
    36. Create an interactive, filterable legend using d3.dispatch() for coordinated views
    37. Implement crosshair functionality for precise data reading
    38. Add a subtle, styled background using d3.select().append("rect") with rounded corners
    39. Ensure the visualization updates smoothly when data changes or on user interaction
    40. Use d3.transition().duration() to control animation speed, with longer durations for more complex animations
    41. Implement staggered animations using d3.transition().delay() to create cascading effects
    42. Utilize d3.easeElastic, d3.easeBack, or custom easing functions for more dynamic animations
    43. Implement enter, update, and exit animations for data changes
    44. Use d3.interpolateString() for smooth transitions between different text values
    45. Implement path animations using d3.interpolate() for custom interpolators
    46. Create looping animations using d3.timer() for continuous effects if applicable
    47. Implement chained transitions using .transition().transition() for sequential animations
    48. Use d3.active() to coordinate multiple animations and prevent overlapping
    49. Implement FLIP (First, Last, Invert, Play) animations for layout changes if applicable
    50. Add dropdowns or other UI elements for users to change the variables displayed on each axis
    51. Implement a brush reset button and functionality
    52. Add a chart title and make it responsive to window resizing
    53. Implement a sophisticated tooltip that shows all relevant information and follows the mouse
    54. Create a legend that allows toggling visibility of different data categories
    55. Implement a size scale for data points based on a specific attribute

    Data Schema:
    {schema_str}

    Sample Data:
    {json.dumps(data_sample[:5], indent=2)}

    IMPORTANT: Your entire response must be valid D3.js code that can be executed directly. Do not include any text before or after the code.
    """

    if user_input:
        prompt = f"""
        # D3.js Code Generation Task

        Generate ONLY D3.js version 7 code for a {viz_type.lower()}, interactive, and comparative visualization. Do not include any explanations, comments, or markdown formatting.

        Critical Requirements:
        1. Create a function named createVisualization(data, svgElement)
        2. Implement a visualization that explicitly compares data from two CSV files AND satisfies this user prompt:
        ---
        {user_input}
        ---
        3. Solve the overlapping labels problem:
           - Rotate labels if necessary (e.g., 45-degree angle)
           - Use a larger SVG size (e.g., width: 1000px, height: 600px) to accommodate all labels
           - Implement label truncation or abbreviation for long names
        4. Use different colors for each data source and implement a color scale
        5. Include an interactive legend clearly indicating which color represents which data source
        6. Ensure appropriate spacing between data points
        7. Add tooltips showing full information on hover
        8. Implement responsive design to fit various screen sizes
        9. Include smooth transitions for any data updates
        10. Implement zoom, pan, and brush functionality
        11. Add dropdowns or other UI elements for users to change the variables displayed on each axis
        12. Implement a brush reset button and functionality
        13. Add a chart title and make it responsive to window resizing
        14. Create a sophisticated tooltip that shows all relevant information and follows the mouse
        15. Implement a size scale for data points based on a specific attribute

        Data Schema:
        {schema_str}

        Sample Data:
        {json.dumps(data_sample[:5], indent=2)}

        Current Code:
        ```javascript
        {st.session_state.current_viz}
        ```

        IMPORTANT: Your entire response must be valid D3.js code that can be executed directly. Do not include any text before or after the code.
        """
    else:
        prompt = base_prompt

    try:
        response = client.chat.completions.create(
            model=st.session_state.get("selected_model", "gpt-4"),  # Use the selected model
            messages=[
                {"role": "system", "content": "You are a D3.js expert specializing in creating sophisticated, interactive, and comparative visualizations. Your code must explicitly address all requirements and ensure a comparative aspect between two data sources."},
                {"role": "user", "content": prompt}
            ]
        )

        d3_code = response.choices[0].message.content
        if not d3_code.strip():
            raise ValueError("Generated D3 code is empty")

        return d3_code
    except Exception as e:
        logger.error(f"Error generating D3 code with OpenAI: {str(e)}")
        return generate_fallback_visualization()

def refine_d3_code_openai(
    initial_code: str,
    api_key: str,
    max_attempts: int = 3
) -> str:
    """
    Refine the D3 code using OpenAI's API.

    Args:
        initial_code (str): The initial D3.js code to refine.
        api_key (str): OpenAI API key.
        max_attempts (int, optional): Maximum number of refinement attempts. Defaults to 3.

    Returns:
        str: Refined D3.js code, or the last attempt if refinement fails.
    """
    client = OpenAI(api_key=api_key)
    
    for attempt in range(max_attempts):
        if validate_d3_code(initial_code):
            return initial_code
        
        refinement_prompt = f"""
        The following D3 code needs refinement to be valid:
        
        {initial_code}
        
        Please provide a corrected version that:
        1. Defines a createVisualization(data, svgElement) function
        2. Uses only D3.js version 7 syntax
        3. Creates a valid visualization
        
        Return ONLY the corrected D3 code without any explanations or comments.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a D3.js expert. Provide only valid D3 code."},
                    {"role": "user", "content": refinement_prompt}
                ]
            )
            
            initial_code = clean_d3_response(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error refining D3 code with OpenAI: {str(e)}")
            break  # Exit the loop if API call fails
    
    logger.warning("Failed to generate valid D3 code after maximum attempts with OpenAI")
    return initial_code

def refine_d3_code_google(
    initial_code: str,
    api_key: str,
    max_attempts: int = 3
) -> str:
    """
    Refine the D3 code using Google's API.

    Args:
        initial_code (str): The initial D3.js code to refine.
        api_key (str): Google API key.
        max_attempts (int, optional): Maximum number of refinement attempts. Defaults to 3.

    Returns:
        str: Refined D3.js code, or the last attempt if refinement fails.
    """
    # Placeholder implementation for Google API refinement
    try:
        for attempt in range(max_attempts):
            if validate_d3_code(initial_code):
                return initial_code

            refinement_prompt = f"""
            The following D3 code needs refinement to be valid:
            
            {initial_code}
            
            Please provide a corrected version that:
            1. Defines a createVisualization(data, svgElement) function
            2. Uses only D3.js version 7 syntax
            3. Creates a valid visualization
            
            Return ONLY the corrected D3 code without any explanations or comments.
            """

            # Implement Google-specific API call for refinement
            # Placeholder response
            refined_code = """
            function createVisualization(data, svgElement) {
                // Refined D3.js code by Google API
                console.log("D3.js code refined using Google API");
            }
            """
            initial_code = clean_d3_response(refined_code)

        logger.warning("Failed to generate valid D3 code after maximum attempts with Google")
        return initial_code
    except Exception as e:
        logger.error(f"Error refining D3 code with Google: {str(e)}")
        return initial_code

def refine_d3_code_anthropic(
    initial_code: str,
    api_key: str,
    max_attempts: int = 3
) -> str:
    """
    Refine the D3 code using Anthropic's API.

    Args:
        initial_code (str): The initial D3.js code to refine.
        api_key (str): Anthropic API key.
        max_attempts (int, optional): Maximum number of refinement attempts. Defaults to 3.

    Returns:
        str: Refined D3.js code, or the last attempt if refinement fails.
    """
    # Placeholder implementation for Anthropic API refinement
    try:
        for attempt in range(max_attempts):
            if validate_d3_code(initial_code):
                return initial_code

            refinement_prompt = f"""
            The following D3 code needs refinement to be valid:
            
            {initial_code}
            
            Please provide a corrected version that:
            1. Defines a createVisualization(data, svgElement) function
            2. Uses only D3.js version 7 syntax
            3. Creates a valid visualization
            
            Return ONLY the corrected D3 code without any explanations or comments.
            """

            # Implement Anthropic-specific API call for refinement
            # Placeholder response
            refined_code = """
            function createVisualization(data, svgElement) {
                // Refined D3.js code by Anthropic API
                console.log("D3.js code refined using Anthropic API");
            }
            """
            initial_code = clean_d3_response(refined_code)

        logger.warning("Failed to generate valid D3 code after maximum attempts with Anthropic")
        return initial_code
    except Exception as e:
        logger.error(f"Error refining D3 code with Anthropic: {str(e)}")
        return initial_code

def clean_d3_response(response: str) -> str:
    """
    Clean the LLM response to ensure it only contains D3 code.
    
    This function removes markdown formatting, non-JavaScript lines,
    and ensures the code starts with the createVisualization function.
    
    Args:
        response (str): The raw response from the LLM.
    
    Returns:
        str: Cleaned D3.js code.
    """
    # Remove any potential markdown code blocks
    response = response.replace("```javascript", "").replace("```", "")
    
    # Remove any lines that don't look like JavaScript
    clean_lines = [line for line in response.split('\n') if line.strip() and not line.strip().startswith('#')]
    
    # Ensure the code starts with the createVisualization function
    if not any(line.strip().startswith('function createVisualization') for line in clean_lines):
        clean_lines.insert(0, 'function createVisualization(data, svgElement) {')
        clean_lines.append('}')
    
    return '\n'.join(clean_lines)

def display_visualization(d3_code: str):
    """
    Display the D3.js visualization using an iframe and add a download button.
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <style>
            #visualization {{
                width: 100%;
                height: 100vh;
                overflow: hidden;
            }}
            svg {{
                width: 100%;
                height: 100%;
            }}
        </style>
    </head>
    <body>
        <div id="visualization"></div>
        <button onclick="downloadSVG()" style="position: absolute; top: 10px; right: 10px;">Download SVG</button>
        <script>
            {d3_code}
            // Create the SVG element
            const svgElement = d3.select("#visualization")
                .append("svg")
                .attr("viewBox", "0 0 960 540")
                .attr("preserveAspectRatio", "xMidYMid meet")
                .node();
            
            // Get the data from the parent window
            const vizData = JSON.parse(decodeURIComponent(window.location.hash.slice(1)));
            
            // Call the createVisualization function
            createVisualization(vizData, svgElement);

            // Function to download the SVG
            function downloadSVG() {{
                const svgData = new XMLSerializer().serializeToString(svgElement);
                const svgBlob = new Blob([svgData], {{type: "image/svg+xml;charset=utf-8"}});
                const svgUrl = URL.createObjectURL(svgBlob);
                const downloadLink = document.createElement("a");
                downloadLink.href = svgUrl;
                downloadLink.download = "visualization.svg";
                document.body.appendChild(downloadLink);
                downloadLink.click();
                document.body.removeChild(downloadLink);
            }}

            // Make the visualization responsive
            window.addEventListener('resize', function() {{
                d3.select(svgElement).attr("viewBox", "0 0 " + window.innerWidth + " " + window.innerHeight);
            }});
        </script>
    </body>
    </html>
    """
    
    # Encode the data to pass it to the iframe
    encoded_data = urllib.parse.quote(json.dumps(st.session_state.preprocessed_df.to_dict(orient='records')))
    
    # Display the iframe with the encoded data in the URL hash
    st.components.v1.iframe(f"data:text/html;charset=utf-8,{urllib.parse.quote(html_content)}#{encoded_data}", 
                            width=960, height=540, scrolling=False)

def generate_fallback_visualization() -> str:
    """
    Generate a fallback visualization if the LLM fails.
    
    This function creates a simple bar chart using D3.js as a fallback
    when the main visualization generation process fails.
    
    Returns:
        str: D3.js code for a simple bar chart visualization.
    """
    logger.info("Generating fallback visualization")
    
    fallback_code = """
    function createVisualization(data, svgElement) {
        const margin = { top: 20, right: 20, bottom: 50, left: 50 };
        const width = 800 - margin.left - margin.right;
        const height = 500 - margin.top - margin.bottom;
        
        svgElement.attr("width", width + margin.left + margin.right)
                   .attr("height", height + margin.top + margin.bottom);
        
        const svg = svgElement.append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);

        // Assuming the first column is for x-axis and second for y-axis
        const xKey = Object.keys(data[0])[0];
        const yKey = Object.keys(data[0])[1];

        const xScale = d3.scaleBand()
            .domain(data.map(d => d[xKey]))
            .range([0, width])
            .padding(0.1);

        const yScale = d3.scaleLinear()
            .domain([0, d3.max(data, d => +d[yKey])])
            .range([height, 0]);

        svg.selectAll("rect")
            .data(data)
            .join("rect")
            .attr("x", d => xScale(d[xKey]))
            .attr("y", d => yScale(+d[yKey]))
            .attr("width", xScale.bandwidth())
            .attr("height", d => height - yScale(+d[yKey]))
            .attr("fill", "steelblue");

        svg.append("g")
            .attr("transform", `translate(0, ${height})`)
            .call(d3.axisBottom(xScale));

        svg.append("g")
            .call(d3.axisLeft(yScale));

        svg.append("text")
            .attr("x", width / 2)
            .attr("y", height + margin.top + 20)
            .attr("text-anchor", "middle")
            .text(xKey);

        svg.append("text")
            .attr("transform", "rotate(-90)")
            .attr("x", -height / 2)
            .attr("y", -margin.left + 20)
            .attr("text-anchor", "middle")
            .text(yKey);
    }
    """
    
    logger.info("Fallback visualization generated successfully")
    return fallback_code

def route_api_call(provider: str, api_key: str, df: pd.DataFrame, user_input: str, viz_type: str) -> str:
    """
    Route API calls to the appropriate provider's implementation.
    """
    provider_functions = {
        "openai": generate_d3_code_openai,
        "google": generate_d3_code_google,
        "anthropic": generate_d3_code_anthropic
    }
    
    if provider.lower() not in provider_functions:
        logger.error(f"Unsupported provider: {provider}")
        return generate_fallback_visualization()
        
    try:
        return provider_functions[provider.lower()](df, api_key, user_input, viz_type)
    except Exception as e:
        logger.error(f"Error generating D3 code with {provider}: {str(e)}")
        return generate_fallback_visualization()

def generate_and_validate_d3_code(
    df: pd.DataFrame,
    api_key: str,
    user_input: str = "",
    viz_type: str = "Bar Chart"
) -> str:
    """
    Generate, validate, and refine D3 code using the selected provider's API.
    """
    provider = st.session_state.get("provider", "openai")
    
    # Generate initial code using the appropriate provider
    initial_code = route_api_call(provider, api_key, df, user_input, viz_type)
    
    # Clean and validate the response
    cleaned_code = clean_d3_response(initial_code)
    
    if validate_d3_code(cleaned_code):
        return cleaned_code
    
    # If validation fails, attempt refinement
    refinement_functions = {
        "openai": refine_d3_code_openai,
        "google": refine_d3_code_google,
        "anthropic": refine_d3_code_anthropic
    }
    
    refine_func = refinement_functions.get(provider, refine_d3_code_openai)
    return refine_func(cleaned_code, api_key)

def get_model_selection() -> str:
    """
    Provide a dropdown menu for users to select a language model and set the provider.
    """
    st.sidebar.header("Model Selection")

    # Define available models per provider
    models = {
        "OpenAI": [
            "gpt-4",
            "gpt-4-0613",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0613",
        ],
        "Google": [
            "PaLM-2",
            "PaLM-2-Chat",
        ],
        "Anthropic": [
            "claude-2",
            "claude-instant-1",
        ],
    }

    # Flatten the models into a list with provider prefixes
    model_options = []
    model_provider_map = {}
    for provider, model_list in models.items():
        for model in model_list:
            display_name = f"{provider}: {model}"
            model_options.append(display_name)
            model_provider_map[model] = provider.lower()

    selected_model = st.sidebar.selectbox(
        "Select a Language Model",
        options=model_options,
        index=0,
    )

    # Extract the provider and model name
    provider, model_name = selected_model.split(": ", 1)
    
    # Store both the model and provider in session state
    st.session_state.selected_model = model_name
    st.session_state.provider = provider.lower()

    return model_name

def display_loading_animation():
    loading_html = """
    <div class="loading-container" style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 500px;">
        <div class="loading-spinner">
            <div class="spinner-ring"></div>
            <div class="spinner-ring"></div>
            <div class="spinner-ring"></div>
            <div class="spinner-ring"></div>
        </div>
        <div class="loading-text">LLM gods are doing magic now...</div>
    </div>
    <style>
        .loading-spinner {
            position: relative;
            width: 80px;
            height: 80px;
        }
        .spinner-ring {
            position: absolute;
            width: 100%;
            height: 100%;
            border: 4px solid transparent;
            border-top-color: #3498db;
            border-radius: 50%;
            animation: spin 1.2s cubic-bezier(0.5, 0, 0.5, 1) infinite;
        }
        .spinner-ring:nth-child(1) { animation-delay: -0.45s; }
        .spinner-ring:nth-child(2) { animation-delay: -0.3s; }
        .spinner-ring:nth-child(3) { animation-delay: -0.15s; }
        .loading-text {
            margin-top: 20px;
            font-family: Arial, sans-serif;
            font-size: 18px;
            color: #3498db;
            animation: pulse 1.5s ease-in-out infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        @keyframes pulse {
            0%, 100% { opacity: 0.5; }
            50% { opacity: 1; }
        }
    </style>
    """
    return st.components.v1.html(loading_html, height=400)

def get_api_credentials() -> Dict[str, str]:
    """
    Retrieve API keys for OpenAI, Google, and Anthropic from Streamlit secrets
    or prompt the user to enter them via sidebar inputs.

    Returns:
        Dict[str, str]: A dictionary containing API keys for each provider.
    """
    api_keys = {
        "openai": st.secrets.get("OPENAI_API_KEY") or st.sidebar.text_input(
            "Enter your OpenAI API Key", type="password"
        ),
        "google": st.secrets.get("GOOGLE_API_KEY") or st.sidebar.text_input(
            "Enter your Google API Key", type="password"
        ),
        "anthropic": st.secrets.get("ANTHROPIC_API_KEY") or st.sidebar.text_input(
            "Enter your Anthropic API Key", type="password"
        ),
    }

    # Display success messages if keys are provided
    for provider, key in api_keys.items():
        if key:
            st.sidebar.success(f"{provider.capitalize()} API key received successfully! 🎉")

    return api_keys

def test_api_key(api_key: str) -> bool:
    """
    Test if the provided API key is valid.
    
    This function attempts to list OpenAI models using the provided API key.
    If successful, the key is considered valid.
    
    Args:
        api_key (str): The OpenAI API key to test.
    
    Returns:
        bool: True if the API key is valid, False otherwise.
    """
    client = OpenAI(api_key=api_key)
    try:
        client.models.list()
        return True
    except Exception as e:
        logger.error(f"API key validation failed: {str(e)}")
        return False

def preprocess_data(file1, file2) -> pd.DataFrame:
    """
    Preprocess and merge the two dataframes for comparison.
    
    This function reads two CSV files, adds a 'Source' column to each,
    merges them, handles missing values, ensures consistent data types,
    and standardizes column names.
    
    Args:
        file1: First CSV file uploaded by the user.
        file2: Second CSV file uploaded by the user.
    
    Returns:
        pd.DataFrame: Preprocessed and merged DataFrame.
    
    Raises:
        ValueError: If files are empty or cannot be parsed.
        Exception: For any other preprocessing errors.
    """
    logger.info("Starting data preprocessing")
    try:
        # Read CSV files into pandas DataFrames
        try:
            df1 = pd.read_csv(file1)
            df2 = pd.read_csv(file2)
        except pd.errors.EmptyDataError:
            raise ValueError("One or both of the uploaded files are empty.")
        except pd.errors.ParserError:
            raise ValueError("Error parsing the CSV files. Please ensure they are valid CSV format.")
        
        # Add 'Source' column to identify the origin of each row
        df1['Source'] = 'CSV file 1'
        df2['Source'] = 'CSV file 2'
        
        # Merge the two DataFrames
        merged_df = pd.concat([df1, df2], ignore_index=True)
        
        # Handle missing values by filling them with 0
        merged_df = merged_df.fillna(0)
        
        # Ensure consistent data types
        for col in merged_df.columns:
            if merged_df[col].dtype == 'object':
                try:
                    merged_df[col] = pd.to_numeric(merged_df[col])
                except ValueError:
                    pass  # Keep as string if can't convert to numeric
        
        # Standardize column names: lowercase and replace spaces with underscores
        merged_df.columns = merged_df.columns.str.lower().str.replace(' ', '_')
        
        logger.info("Data preprocessing completed successfully")
        return merged_df
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        raise

def validate_d3_code(code: str) -> bool:
    """
    Perform basic validation on the generated D3 code.
    
    This function checks for the presence of key D3.js elements and syntax.
    
    Args:
        code (str): The D3.js code to validate.
    
    Returns:
        bool: True if the code passes basic validation, False otherwise.
    """
    # Check if the code defines the createVisualization function
    if not re.search(r'function\s+createVisualization\s*\(data,\s*svgElement\)\s*{', code):
        return False
    
    # Check for basic D3 v7 method calls
    d3_methods = ['d3.select', 'd3.scaleLinear', 'd3.axisBottom', 'd3.axisLeft']
    if not any(method in code for method in d3_methods):
        return False
    
    # Check for balanced braces
    if code.count('{') != code.count('}'):
        return False
    
    return True

def generate_d3_code_openai(
    df: pd.DataFrame,
    api_key: str,
    user_input: str = "",
    viz_type: str = "Bar Chart"
) -> str:
    """
    Generate D3.js code using OpenAI's API.

    Args:
        df (pd.DataFrame): The preprocessed DataFrame.
        api_key (str): OpenAI API key.
        user_input (str, optional): Additional user requirements for visualization.
        viz_type (str, optional): Type of visualization to generate.

    Returns:
        str: Generated D3.js code.
    """
    logger.info("Starting D3 code generation with OpenAI")
    data_sample = df.head(5).to_dict(orient='records')
    schema = df.dtypes.to_dict()
    schema_str = "\n".join([f"{col}: {dtype}" for col, dtype in schema.items()])

    client = OpenAI(api_key=api_key)  # Assuming OpenAI SDK is used

    base_prompt = f"""
    # D3.js Code Generation Task

    Generate ONLY D3.js version 7 code for a {viz_type.lower()}, interactive, and comparative visualization. Do not include any explanations, comments, or markdown formatting.

    Critical Requirements:
    1. Create a function named createVisualization(data, svgElement)
    2. Implement a responsive SVG that adjusts its size based on the content
    3. Utilize the full width and height provided (1000x600 pixels)
    4. Implement zoom, pan, and brush functionality for exploring the data
    5. Ensure efficient use of space, minimizing empty areas
    6. Add appropriate margins, title, axes labels, and an interactive legend
    7. Use different colors for each data source and implement a color scale
    8. Implement tooltips showing full information on hover
    9. Include smooth transitions for any data updates or interactions
    10. Handle potential overlapping of data points or labels
    11. Always have grid lines on the visualization
    12. Animate the visualization as much as possible
    13. Utilize d3.select() for DOM manipulation and d3.data() for data binding
    14. Implement advanced scales: d3.scaleLinear(), d3.scaleBand(), d3.scaleTime(), d3.scaleOrdinal(d3.schemeCategory10)
    15. Create dynamic, animated axes using d3.axisBottom(), d3.axisLeft() with custom tick formatting
    16. Implement smooth transitions and animations using d3.transition() and d3.easeCubic
    17. Utilize d3.line(), d3.area(), d3.arc() for creating complex shapes and paths
    18. Implement interactivity: d3.brush(), d3.zoom(), d3.drag() for user interaction
    19. Use d3.interpolate() for smooth color and value transitions
    20. Implement d3.forceSimulation() for force-directed graph layouts if applicable
    21. Utilize d3.geoPath() and d3.geoProjection() for geographical visualizations if applicable
    22. Use d3.contours() and d3.density2D() for density and contour visualizations if applicable
    23. Implement d3.voronoi() for proximity-based visualizations if applicable
    24. Utilize d3.chord() and d3.ribbon() for relationship visualizations if applicable
    25. Implement advanced event handling with d3.on() for mouseover, click, etc.
    26. Use d3.format() for number formatting in tooltips and labels
    27. Implement d3.timeFormat() for date/time formatting if applicable
    28. Utilize d3.range() and d3.shuffle() for data generation and randomization if needed
    29. Implement d3.nest() for data restructuring and aggregation if needed
    30. Use d3.queue() for asynchronous data loading and processing
    31. Implement accessibility features using ARIA attributes
    32. Optimize performance using d3.quadtree() for spatial indexing if applicable
    33. Implement responsive design using d3.select(window).on("resize", ...)
    34. Focus on creating a comparative visualization that highlights data differences
    35. Implement error handling for invalid data formats and gracefully handle missing data
    36. Create an interactive, filterable legend using d3.dispatch() for coordinated views
    37. Implement crosshair functionality for precise data reading
    38. Add a subtle, styled background using d3.select().append("rect") with rounded corners
    39. Ensure the visualization updates smoothly when data changes or on user interaction
    40. Use d3.transition().duration() to control animation speed, with longer durations for more complex animations
    41. Implement staggered animations using d3.transition().delay() to create cascading effects
    42. Utilize d3.easeElastic, d3.easeBack, or custom easing functions for more dynamic animations
    43. Implement enter, update, and exit animations for data changes
    44. Use d3.interpolateString() for smooth transitions between different text values
    45. Implement path animations using d3.interpolate() for custom interpolators
    46. Create looping animations using d3.timer() for continuous effects if applicable
    47. Implement chained transitions using .transition().transition() for sequential animations
    48. Use d3.active() to coordinate multiple animations and prevent overlapping
    49. Implement FLIP (First, Last, Invert, Play) animations for layout changes if applicable
    50. Add dropdowns or other UI elements for users to change the variables displayed on each axis
    51. Implement a brush reset button and functionality
    52. Add a chart title and make it responsive to window resizing
    53. Implement a sophisticated tooltip that shows all relevant information and follows the mouse
    54. Create a legend that allows toggling visibility of different data categories
    55. Implement a size scale for data points based on a specific attribute

    Data Schema:
    {schema_str}

    Sample Data:
    {json.dumps(data_sample[:5], indent=2)}

    IMPORTANT: Your entire response must be valid D3.js code that can be executed directly. Do not include any text before or after the code.
    """

    if user_input:
        prompt = f"""
        # D3.js Code Generation Task

        Generate ONLY D3.js version 7 code for a {viz_type.lower()}, interactive, and comparative visualization. Do not include any explanations, comments, or markdown formatting.

        Critical Requirements:
        1. Create a function named createVisualization(data, svgElement)
        2. Implement a visualization that explicitly compares data from two CSV files AND satisfies this user prompt:
        ---
        {user_input}
        ---
        3. Solve the overlapping labels problem:
           - Rotate labels if necessary (e.g., 45-degree angle)
           - Use a larger SVG size (e.g., width: 1000px, height: 600px) to accommodate all labels
           - Implement label truncation or abbreviation for long names
        4. Use different colors for each data source and implement a color scale
        5. Include an interactive legend clearly indicating which color represents which data source
        6. Ensure appropriate spacing between data points
        7. Add tooltips showing full information on hover
        8. Implement responsive design to fit various screen sizes
        9. Include smooth transitions for any data updates
        10. Implement zoom, pan, and brush functionality
        11. Add dropdowns or other UI elements for users to change the variables displayed on each axis
        12. Implement a brush reset button and functionality
        13. Add a chart title and make it responsive to window resizing
        14. Create a sophisticated tooltip that shows all relevant information and follows the mouse
        15. Implement a size scale for data points based on a specific attribute

        Data Schema:
        {schema_str}

        Sample Data:
        {json.dumps(data_sample[:5], indent=2)}

        Current Code:
        ```javascript
        {st.session_state.current_viz}
        ```

        IMPORTANT: Your entire response must be valid D3.js code that can be executed directly. Do not include any text before or after the code.
        """
    else:
        prompt = base_prompt

    try:
        response = client.chat.completions.create(
            model=st.session_state.get("selected_model", "gpt-4"),  # Use the selected model
            messages=[
                {"role": "system", "content": "You are a D3.js expert specializing in creating sophisticated, interactive, and comparative visualizations. Your code must explicitly address all requirements and ensure a comparative aspect between two data sources."},
                {"role": "user", "content": prompt}
            ]
        )

        d3_code = response.choices[0].message.content
        if not d3_code.strip():
            raise ValueError("Generated D3 code is empty")

        return d3_code
    except Exception as e:
        logger.error(f"Error generating D3 code with OpenAI: {str(e)}")
        return generate_fallback_visualization()

def refine_d3_code_openai(
    initial_code: str,
    api_key: str,
    max_attempts: int = 3
) -> str:
    """
    Refine the D3 code using OpenAI's API.

    Args:
        initial_code (str): The initial D3.js code to refine.
        api_key (str): OpenAI API key.
        max_attempts (int, optional): Maximum number of refinement attempts. Defaults to 3.

    Returns:
        str: Refined D3.js code, or the last attempt if refinement fails.
    """
    client = OpenAI(api_key=api_key)
    
    for attempt in range(max_attempts):
        if validate_d3_code(initial_code):
            return initial_code
        
        refinement_prompt = f"""
        The following D3 code needs refinement to be valid:
        
        {initial_code}
        
        Please provide a corrected version that:
        1. Defines a createVisualization(data, svgElement) function
        2. Uses only D3.js version 7 syntax
        3. Creates a valid visualization
        
        Return ONLY the corrected D3 code without any explanations or comments.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a D3.js expert. Provide only valid D3 code."},
                    {"role": "user", "content": refinement_prompt}
                ]
            )
            
            initial_code = clean_d3_response(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error refining D3 code with OpenAI: {str(e)}")
            break  # Exit the loop if API call fails
    
    logger.warning("Failed to generate valid D3 code after maximum attempts with OpenAI")
    return initial_code

def refine_d3_code_google(
    initial_code: str,
    api_key: str,
    max_attempts: int = 3
) -> str:
    """
    Refine the D3 code using Google's API.

    Args:
        initial_code (str): The initial D3.js code to refine.
        api_key (str): Google API key.
        max_attempts (int, optional): Maximum number of refinement attempts. Defaults to 3.

    Returns:
        str: Refined D3.js code, or the last attempt if refinement fails.
    """
    # Placeholder implementation for Google API refinement
    try:
        for attempt in range(max_attempts):
            if validate_d3_code(initial_code):
                return initial_code

            refinement_prompt = f"""
            The following D3 code needs refinement to be valid:
            
            {initial_code}
            
            Please provide a corrected version that:
            1. Defines a createVisualization(data, svgElement) function
            2. Uses only D3.js version 7 syntax
            3. Creates a valid visualization
            
            Return ONLY the corrected D3 code without any explanations or comments.
            """

            # Implement Google-specific API call for refinement
            # Placeholder response
            refined_code = """
            function createVisualization(data, svgElement) {
                // Refined D3.js code by Google API
                console.log("D3.js code refined using Google API");
            }
            """
            initial_code = clean_d3_response(refined_code)

        logger.warning("Failed to generate valid D3 code after maximum attempts with Google")
        return initial_code
    except Exception as e:
        logger.error(f"Error refining D3 code with Google: {str(e)}")
        return initial_code

def refine_d3_code_anthropic(
    initial_code: str,
    api_key: str,
    max_attempts: int = 3
) -> str:
    """
    Refine the D3 code using Anthropic's API.

    Args:
        initial_code (str): The initial D3.js code to refine.
        api_key (str): Anthropic API key.
        max_attempts (int, optional): Maximum number of refinement attempts. Defaults to 3.

    Returns:
        str: Refined D3.js code, or the last attempt if refinement fails.
    """
    # Placeholder implementation for Anthropic API refinement
    try:
        for attempt in range(max_attempts):
            if validate_d3_code(initial_code):
                return initial_code

            refinement_prompt = f"""
            The following D3 code needs refinement to be valid:
            
            {initial_code}
            
            Please provide a corrected version that:
            1. Defines a createVisualization(data, svgElement) function
            2. Uses only D3.js version 7 syntax
            3. Creates a valid visualization
            
            Return ONLY the corrected D3 code without any explanations or comments.
            """

            # Implement Anthropic-specific API call for refinement
            # Placeholder response
            refined_code = """
            function createVisualization(data, svgElement) {
                // Refined D3.js code by Anthropic API
                console.log("D3.js code refined using Anthropic API");
            }
            """
            initial_code = clean_d3_response(refined_code)

        logger.warning("Failed to generate valid D3 code after maximum attempts with Anthropic")
        return initial_code
    except Exception as e:
        logger.error(f"Error refining D3 code with Anthropic: {str(e)}")
        return initial_code

def clean_d3_response(response: str) -> str:
    """
    Clean the LLM response to ensure it only contains D3 code.
    
    This function removes markdown formatting, non-JavaScript lines,
    and ensures the code starts with the createVisualization function.
    
    Args:
        response (str): The raw response from the LLM.
    
    Returns:
        str: Cleaned D3.js code.
    """
    # Remove any potential markdown code blocks
    response = response.replace("```javascript", "").replace("```", "")
    
    # Remove any lines that don't look like JavaScript
    clean_lines = [line for line in response.split('\n') if line.strip() and not line.strip().startswith('#')]
    
    # Ensure the code starts with the createVisualization function
    if not any(line.strip().startswith('function createVisualization') for line in clean_lines):
        clean_lines.insert(0, 'function createVisualization(data, svgElement) {')
        clean_lines.append('}')
    
    return '\n'.join(clean_lines)

def display_visualization(d3_code: str):
    """
    Display the D3.js visualization using an iframe and add a download button.
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <style>
            #visualization {{
                width: 100%;
                height: 100vh;
                overflow: hidden;
            }}
            svg {{
                width: 100%;
                height: 100%;
            }}
        </style>
    </head>
    <body>
        <div id="visualization"></div>
        <button onclick="downloadSVG()" style="position: absolute; top: 10px; right: 10px;">Download SVG</button>
        <script>
            {d3_code}
            // Create the SVG element
            const svgElement = d3.select("#visualization")
                .append("svg")
                .attr("viewBox", "0 0 960 540")
                .attr("preserveAspectRatio", "xMidYMid meet")
                .node();
            
            // Get the data from the parent window
            const vizData = JSON.parse(decodeURIComponent(window.location.hash.slice(1)));
            
            // Call the createVisualization function
            createVisualization(vizData, svgElement);

            // Function to download the SVG
            function downloadSVG() {{
                const svgData = new XMLSerializer().serializeToString(svgElement);
                const svgBlob = new Blob([svgData], {{type: "image/svg+xml;charset=utf-8"}});
                const svgUrl = URL.createObjectURL(svgBlob);
                const downloadLink = document.createElement("a");
                downloadLink.href = svgUrl;
                downloadLink.download = "visualization.svg";
                document.body.appendChild(downloadLink);
                downloadLink.click();
                document.body.removeChild(downloadLink);
            }}

            // Make the visualization responsive
            window.addEventListener('resize', function() {{
                d3.select(svgElement).attr("viewBox", "0 0 " + window.innerWidth + " " + window.innerHeight);
            }});
        </script>
    </body>
    </html>
    """
    
    # Encode the data to pass it to the iframe
    encoded_data = urllib.parse.quote(json.dumps(st.session_state.preprocessed_df.to_dict(orient='records')))
    
    # Display the iframe with the encoded data in the URL hash
    st.components.v1.iframe(f"data:text/html;charset=utf-8,{urllib.parse.quote(html_content)}#{encoded_data}", 
                            width=960, height=540, scrolling=False)

def generate_fallback_visualization() -> str:
    """
    Generate a fallback visualization if the LLM fails.
    
    This function creates a simple bar chart using D3.js as a fallback
    when the main visualization generation process fails.
    
    Returns:
        str: D3.js code for a simple bar chart visualization.
    """
    logger.info("Generating fallback visualization")
    
    fallback_code = """
    function createVisualization(data, svgElement) {
        const margin = { top: 20, right: 20, bottom: 50, left: 50 };
        const width = 800 - margin.left - margin.right;
        const height = 500 - margin.top - margin.bottom;
        
        svgElement.attr("width", width + margin.left + margin.right)
                   .attr("height", height + margin.top + margin.bottom);
        
        const svg = svgElement.append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);

        // Assuming the first column is for x-axis and second for y-axis
        const xKey = Object.keys(data[0])[0];
        const yKey = Object.keys(data[0])[1];

        const xScale = d3.scaleBand()
            .domain(data.map(d => d[xKey]))
            .range([0, width])
            .padding(0.1);

        const yScale = d3.scaleLinear()
            .domain([0, d3.max(data, d => +d[yKey])])
            .range([height, 0]);

        svg.selectAll("rect")
            .data(data)
            .join("rect")
            .attr("x", d => xScale(d[xKey]))
            .attr("y", d => yScale(+d[yKey]))
            .attr("width", xScale.bandwidth())
            .attr("height", d => height - yScale(+d[yKey]))
            .attr("fill", "steelblue");

        svg.append("g")
            .attr("transform", `translate(0, ${height})`)
            .call(d3.axisBottom(xScale));

        svg.append("g")
            .call(d3.axisLeft(yScale));

        svg.append("text")
            .attr("x", width / 2)
            .attr("y", height + margin.top + 20)
            .attr("text-anchor", "middle")
            .text(xKey);

        svg.append("text")
            .attr("transform", "rotate(-90)")
            .attr("x", -height / 2)
            .attr("y", -margin.left + 20)
            .attr("text-anchor", "middle")
            .text(yKey);
    }
    """
    
    logger.info("Fallback visualization generated successfully")
    return fallback_code

def main():
    st.set_page_config(
        page_title="🎨 Comparative Visualization Generator",
        page_icon="✨",
        layout="wide",
    )
    st.title("🎨 Comparative Visualization Generator")

    # Retrieve API credentials
    api_credentials = get_api_credentials()

    # Model Selection
    selected_model = get_model_selection()

    # Determine the provider of the selected model
    provider = st.session_state.get("provider")
    if not provider:
        st.error("Selected model provider is not recognized.")
        st.stop()

    # Retrieve the corresponding API key
    api_key = api_credentials.get(provider)
    if not api_key:
        st.error(f"API key for {provider.capitalize()} is missing. Please provide it in the sidebar.")
        st.stop()

    st.header("Upload CSV Files")
    col1, col2 = st.columns(2)
    with col1:
        file1 = st.file_uploader("Upload first CSV file", type="csv")
    with col2:
        file2 = st.file_uploader("Upload second CSV file", type="csv")

    if 'update_viz' not in st.session_state:
        st.session_state.update_viz = False

    if file1 and file2:
        try:
            if (
                'preprocessed_df' not in st.session_state
                or st.session_state.preprocessed_df is None
                or st.session_state.update_viz
            ):
                with st.spinner("Preprocessing data..."):
                    merged_df = preprocess_data(file1, file2)
                st.session_state.preprocessed_df = merged_df
                st.session_state.update_viz = False  # Reset the flag

            with st.expander("Preview of preprocessed data"):
                st.dataframe(st.session_state.preprocessed_df.head())

            if 'current_viz' not in st.session_state or st.session_state.current_viz is None:
                with st.spinner("Generating initial D3 visualization..."):
                    d3_code = generate_and_validate_d3_code(
                        st.session_state.preprocessed_df,
                        api_key,
                        user_input="",  # Initial request
                        viz_type=st.session_state.get("viz_type", "Bar Chart")
                    )
                    st.session_state.current_viz = d3_code
                    st.session_state.workflow_history.append({
                        "version": len(st.session_state.workflow_history) + 1,
                        "request": "Initial comparative visualization",
                        "code": d3_code
                    })

            # Create a placeholder for the visualization
            viz_placeholder = st.empty()

            # Display the current visualization
            with viz_placeholder.container():
                st.subheader("Current Visualization")
                display_visualization(st.session_state.current_viz)

            st.subheader("Modify Visualization")
            user_input = st.text_area(
                "Enter your modification request (or type 'exit' to finish):",
                height=100
            )

            if st.button("Update Visualization"):
                if user_input.lower().strip() == 'exit':
                    st.success("Visualization process completed.")
                elif user_input:
                    # Replace current visualization with loading animation
                    with viz_placeholder.container():
                        st.subheader("Updating Visualization")
                        display_loading_animation()

                    # Generate new visualization
                    with st.spinner("Generating updated visualization..."):
                        modified_d3_code = generate_and_validate_d3_code(
                            st.session_state.preprocessed_df,
                            api_key,
                            user_input,
                            viz_type=st.session_state.get("viz_type", "Bar Chart")
                        )
                        st.session_state.current_viz = modified_d3_code
                        st.session_state.workflow_history.append({
                            "version": len(st.session_state.workflow_history) + 1,
                            "request": user_input,
                            "code": modified_d3_code
                        })

                    # Update the visualization in place
                    with viz_placeholder.container():
                        st.subheader("Current Visualization")
                        display_visualization(st.session_state.current_viz)
                else:
                    st.warning("Please enter a modification request or type 'exit' to finish.")

            with st.expander("View/Edit Visualization Code"):
                code_editor = st.text_area(
                    "D3.js Code",
                    value=st.session_state.current_viz,
                    height=300,
                    key="code_editor"
                )
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    edit_enabled = st.checkbox("Enable Edit", key="edit_toggle")
                with col2:
                    if st.button("Execute Code"):
                        if edit_enabled:
                            if validate_d3_code(code_editor):
                                st.session_state.current_viz = code_editor
                                st.session_state.workflow_history.append({
                                    "request": "Manual code edit",
                                    "code": code_editor
                                })
                                if len(st.session_state.workflow_history) > MAX_WORKFLOW_HISTORY:
                                    st.session_state.workflow_history.pop(0)
                                # Update the visualization in place
                                with viz_placeholder.container():
                                    st.subheader("Current Visualization")
                                    display_visualization(st.session_state.current_viz)
                            else:
                                st.error("Invalid D3.js code. Please check your code and try again.")
                        else:
                            st.warning("Enable 'Edit' to make changes.")
                with col3:
                    if st.button("Copy Code"):
                        st.write("Code copied to clipboard!")
                        st.write(f'<textarea style="position: absolute; left: -9999px;">{code_editor}</textarea>', unsafe_allow_html=True)
                        st.write('<script>document.querySelector("textarea").select();document.execCommand("copy");</script>', unsafe_allow_html=True)

            with st.expander("Workflow History"):
                for i, step in enumerate(st.session_state.workflow_history):
                    st.subheader(f"Step {i+1}")
                    st.write(f"Request: {step['request']}")
                    if st.button(f"Revert to Step {i+1}"):
                        st.session_state.current_viz = step['code']
                        # Update the visualization in place
                        with viz_placeholder.container():
                            st.subheader("Current Visualization")
                            display_visualization(st.session_state.current_viz)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"Error in main function: {str(e)}")
            logger.error(traceback.format_exc())
            st.error("An unexpected error occurred. Please try again or contact support if the problem persists.")
            st.code(traceback.format_exc())  # Display traceback for debugging
    else:
        st.info("Please upload both CSV files to visualize your data")

if __name__ == "__main__":
    main()
    
    #add a button to download the svg 