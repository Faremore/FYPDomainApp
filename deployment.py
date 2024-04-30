import streamlit as st
import pandas as pd
import requests
pip install python-whois
import whois
from datetime import datetime

api_key = 'AIzaSyB9ESjm8yiiyA4Fkr386lAoc3FhyAQ1G7M'

def get_seo_score(url):
    api_url = f"https://www.googleapis.com/pagespeedonline/v5/runPagespeed?url={url}&key={api_key}&category=SEO"
    response = requests.get(api_url)
    result = response.json()

    try:
        seo_score = result['lighthouseResult']['categories']['seo']['score'] * 100
    except KeyError:
        # Handle missing data in the response
        # print(f"Error retrieving SEO score for {url}.")
        return 0  # or you can return a default score or message

    return seo_score

def get_domain_age(domain):
    try:
        # Perform a WHOIS query to get domain information
        domain_info = whois.whois(domain)
        # Extract the creation date
        creation_date = domain_info.creation_date
        if isinstance(creation_date, list):  # Handle cases where multiple dates are returned
            creation_date = creation_date[0]
        # Calculate the age of the domain
        current_date = datetime.now()
        age_years = current_date.year - creation_date.year - ((current_date.month, current_date.day) < (creation_date.month, creation_date.day))
        return age_years
    except Exception as e:
        return 0

# TLD and score mapping
tld_scores = {
    '.com': 36,
    '.cn': 4,
    '.tk': 3,
    '.de': 3,
    '.uk': 3,
    '.net': 3,
    '.org': 2,
    '.xyz': 1,
    '.ru': 1,
    '.top': 1,
}

# Function to calculate the score of a list of domain names
def get_TLD_score(domain):
    domain_parts = domain.split('.')

    # Prioritize checking for second-level TLDs accurately
    if len(domain_parts) >= 3:
        potential_tld = '.' + domain_parts[-2] + '.' + domain_parts[-1]
        if potential_tld in tld_scores:
            tld = potential_tld
        else:
            tld = '.' + domain_parts[-1]  # Fallback
    else:
        tld = '.' + domain_parts[-1]  # Only one TLD part

    score = tld_scores.get(tld, 1)  # Get score, default to 1

    # Rescale the score to a maximum of 100
    max_score = 100
    rescaled_score = (score / max(tld_scores.values())) * max_score

    return round(rescaled_score, 2)

# Google Custom Search API key
api_keyJR = 'AIzaSyB9ESjm8yiiyA4Fkr386lAoc3FhyAQ1G7M'
# Custom Search Engine ID
cse_id = '71e10fbd14100465b'

import time
def get_search_queries(query):
    # Extract domain name without TLD
    domain_parts = query.split('.')
    domain_without_tld = '.'.join(domain_parts[:-1])  # join all parts except the last TLD part

    url = f'https://www.googleapis.com/customsearch/v1?q={domain_without_tld}&cx={cse_id}&key={api_keyJR}'
    response = requests.get(url)
    results = response.json()
    time.sleep(0.35)
    # Handle cases where 'searchInformation' is not present in the response
    if 'searchInformation' not in results:
        print(f"Error or no results for query '{domain_without_tld}': {results.get('error',{})}")
        return 0

    # Capture the total number of search results
    total_results = results['searchInformation']['totalResults']
    return total_results

import spacy
import pandas as pd
import re
from wordsegment import load, segment

# Initialize wordsegment and spaCy
load()
nlp = spacy.load("en_core_web_lg")

# Load the local dictionary from a text file
with open('words.txt', 'r') as file:
    local_dictionary = set(word.strip().lower() for word in file)

meaningful_short_forms = {'ai', 'ui', 'ux'}

def is_word_meaningful(word):
    return word.lower() in local_dictionary

def is_possible_name(word):
    doc = nlp(word)
    return any(ent.label_ in ["PERSON", "ORG"] for ent in doc.ents)

def calculate_score(meaningful_words, domain):
    # Calculate the total length of all meaningful words
    meaningful_length = sum(len(word) for word in meaningful_words.split())
    # Calculate the length of the domain's main part (excluding TLD, 'www', and non-alphanumeric characters)
    domain_main_part = re.sub(r'[^a-zA-Z0-9]', '', domain.replace('www.', ''))
    domain_main_part_length = len(domain_main_part)
    # Calculate the score as a percentage
    score = (meaningful_length / domain_main_part_length) * 100 if domain_main_part_length else 0
    # Return the score formatted to two decimal places
    return round(score, 2)


def get_preprocess_domain(domain):
    domain = domain.lower().strip().rstrip('.')
    domain = domain.split("www.")[-1]
    domain_main_part, sep, tld = domain.rpartition('.')

    # Normalize the domain name by removing non-alphanumeric characters except for the hyphen
    domain_normalized = re.sub(r'[^a-zA-Z0-9-]', '', domain_main_part)

    # Special handling for 'ly' TLD
    if tld == "ly":
        domain_normalized = domain_normalized.replace("-", " ")  # Replace hyphens with spaces for segmentation
        segmented = segment(domain_normalized)
        if segmented:
            last_segment = segmented[-1]  # Get the last segment
            combined_word = last_segment + tld  # Combine it with 'ly'
            # Check if the combined word is meaningful
            if is_word_meaningful(combined_word):
                # If meaningful, replace the last segment with the combined form
                segmented[-1] = combined_word
                domain_main_part = ' '.join(segmented)  # Update the domain main part
                tld = ""  # Reset TLD as the combined word is now part of the main part

    # Filter segments based on meaningfulness and exclude single-letter segments that are not predefined as meaningful
    segmented = segment(domain_main_part.replace("-", ""))  # Remove hyphens before segmentation

    meaningful_segments = [
        seg for seg in segmented
        if is_word_meaningful(seg) and (len(seg) > 1 or seg in meaningful_short_forms)
    ]
    meaningful_words = ' '.join(meaningful_segments)

    # Remove non-alphanumeric characters for the word count comparison
    domain_alpha_only = re.sub(r'[^a-zA-Z0-9]', '', domain_main_part)

    # Calculate the score
    score = calculate_score(meaningful_words, domain_alpha_only)
    # return meaningful_words, len(meaningful_segments), score  # Check the Meaningful word, total of meaning word, and score
    return score  # Only return score

def get_length_score(domain):
    # Lowercase and strip any trailing periods
    domain = domain.lower().rstrip('.')

    # Split by '.' and remove 'www' if it is a subdomain
    parts = domain.split('.')
    parts = [part for part in parts if part != 'www']

    # The main part of the domain is now the first element in the list `parts`
    # We exclude the last part since it's the TLD
    main_part = parts[0] if len(parts) > 1 else ''

    # Count the length of the main part of the domain
    length = len(main_part)

    # Assign a score based on the length tier
    if length <= 5:
        return 100  # Tier 1: Short
    elif length <= 10:
        return 50   # Tier 2: Medium
    else:
        return 10   # Tier 3: Long

import requests
import pandas as pd
import re
from wordsegment import load, segment

# Initialize the wordsegment module
load()
api_token = "9Sl6auAz"

def search_company_name(name, api_token):
    base_url = "https://api.thecompaniesapi.com/v1/companies/by-name"
    query_params = {"name": name, "token": api_token}
    response = requests.get(base_url, params=query_params)
    try:
        response.raise_for_status()
        data = response.json()
        return bool(data['companies'])  # Assuming if companies is non-empty, the brand exists
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error for {name}: {e}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"Request exception for {name}: {e}")
        return False

def get_brand_length_score(brand):
    length = len(brand)
    if length <= 5:
        return 100
    elif length <= 10:
        return 50
    else:
        return 10

def get_brand_score(domain):
    domain = domain.lower().strip().rstrip('.').split('//')[-1].split('www.')[-1]
    main_part, _, _ = domain.rpartition('.')
    main_part_cleaned = main_part.replace('-', '')

    segmented = segment(main_part_cleaned)
    brand_names = [seg for seg in segmented if search_company_name(seg, api_token)]

    # If there's only one brand name found or the domain itself is a single segment, apply logic accordingly
    if len(segmented) == 1:
        if brand_names:
            # If the single segment is a recognized brand, apply the score and the 1.5 multiplier directly
            total_score = get_brand_length_score(segmented[0]) * 1.5
        else:
            # If the single segment is not a recognized brand, it gets the base score without multiplier
            total_score = get_brand_length_score(segmented[0])
    else:
        total_score = 0
        is_all_segments_brand = len(segmented) == len(brand_names)

        # Score each brand based on its position
        for i, brand in enumerate(brand_names):
            brand_score = get_brand_length_score(brand)
            if i == 0:  # First segment
                brand_score *= 1.3
            elif i == len(brand_names) - 1 and len(brand_names) > 1:  # Last segment
                brand_score *= 1.1
            else:  # Middle segments
                brand_score *= 1.2
            total_score += brand_score

        # Apply a multiplier if all segments are brand names
        if is_all_segments_brand:
            total_score *= 1.5

    return total_score

def calculate_factors(domain):

    # Construct the output dictionary
    output = {
        "Name": domain,
        "SEO Ranking":  get_seo_score(domain),
        "Domain Age": get_domain_age(domain),
        "TLD Popularity Share": get_TLD_score(domain),
        "Search Results Count": get_search_queries(domain),
        "Name Composition": get_preprocess_domain(domain),
        "Length Score": get_length_score(domain),
        "Brand Score" :get_brand_score(domain)
    }

    return output

def extra_features(domain, output):
    # Create a DataFrame with the input domain
    df = pd.DataFrame({'Domain Name': [domain]})

    # Remove .com, .net, etc from the domain name
    df['Domain_Name_Split'] = df['Domain Name'].str.split('.', expand=True)[0]

    # Extract lengths of the domain names
    df['Length'] = df['Domain_Name_Split'].astype(str).str.len()

    # Extract special characters from the domains
    df["Special_chars"] = df.apply(lambda p: sum(not q.isalpha() and not q.isdigit() for q in p["Domain_Name_Split"]), axis=1)

    # Extract letters and their properties from the domains including Uppercase and Lowercase
    df["Letters"] = df.apply(lambda p: sum(q.isalpha() for q in p["Domain_Name_Split"]), axis=1)
    df['Vowels'] = df['Domain_Name_Split'].str.lower().str.count(r'[aeiou]')
    df['Consonants'] = df['Domain_Name_Split'].str.lower().str.count(r'[a-z]') - df['Vowels']

    # Extract numbers from the domains
    df["Numbers"] = df.apply(lambda p: sum(q.isdigit() for q in p["Domain_Name_Split"]), axis=1)

    # Construct dictionary
    extrafeatures = {
        "Length": int(df['Length'].values[0]),
        "Special_chars": int(df["Special_chars"].values[0]),
        "Letters": int(df["Letters"].values[0]),
        "Vowels": int(df["Vowels"].values[0]),
        "Consonants": int(df["Consonants"].values[0]),
        "Numbers": int(df["Numbers"].values[0])
    }
    # Append the DataFrame to the output dictionary
    output = output.update(extrafeatures)
    return output

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def calculate_price(output, feature_weightages):
    # Define numerical columns
    numerical_columns = ['Domain Age', 'TLD Popularity Share', 'Search Results Count', 'Name Composition', 'Length', 'Special_chars', 'Letters', 'Vowels', 'Consonants', 'Numbers']

    # Define the data ranges for each feature
    feature_ranges = {
        'Domain Age': (0, 31),
        'TLD Popularity Share': (1, 100),
        'Search Results Count': (0, 63180000000),
        'Name Composition': (0, 100),
        'Length': (1, 56),
        'Special_chars': (0, 6),
        'Letters': (0, 56),
        'Vowels': (0, 24),
        'Consonants': (0, 33),
        'Numbers': (0, 29)
    }

    # Create a dictionary to store scalers for each feature
    scalers = {}

    # Create and fit scalers for each feature
    for feature in numerical_columns:
        min_val, max_val = feature_ranges[feature]
        scaler = MinMaxScaler(feature_range=(1, 2))
        scaler.fit([[min_val], [max_val]])
        scalers[feature] = scaler

    # Convert the output dictionary to a DataFrame
    temp = pd.DataFrame([output])

    # Convert numerical columns to float
    for col in numerical_columns:
        try:
            temp[col] = temp[col].astype(float)
        except ValueError:
            # If conversion fails, try to extract numeric part from string
            temp[col] = temp[col].str.extract('(\d+)').astype(float)

    # Scale numerical columns to the range
    for feature in numerical_columns:
        scaler = scalers[feature]
        temp[feature] = scaler.transform([[temp[feature].values[0]]])[0][0]

    # Compute weighted score for each domain
    temp['Score'] = sum(temp[feature] * feature_weightages[feature] for feature in feature_weightages.keys() if feature in numerical_columns)

    # Define the original and target ranges
    original_range = [-11.789579, 0]
    target_range = (4.369, 10)

    # Create a MinMaxScaler object for scaling
    scaler = MinMaxScaler(feature_range=target_range)

    # Fit the scaler to the original range
    scaler.fit([[original_range[0]], [original_range[1]]])

    # Scale the score to the target range
    temp['scaled_score'] = scaler.transform([[temp['Score'].values[0]]])[0][0]  
    temp['Estimated Price'] = np.exp(temp['scaled_score'])
    return temp[['Name', 'Estimated Price']]

feature_weightages = {
    'Domain Age': 1.5042875713238217,
    'TLD Popularity Share': 1.9393637880874002,
    'Name Composition': 0.52499056010289487,
    'Length Score': -0.019658760933614740,
    'Search Results Count': 0.9438294316381022,
    'Special_chars': -1.5246809357131826,
    'Consonants': -2.2657756885381935,
    'Letters': -1.1075215834220573,
    'Vowels': 0.01755684381913594,
    'Length': -2.9175410221075556,
    'Numbers': -1.8398622908028888
}
# Create the sidebar
with st.sidebar:
    # Add UI elements to the sidebar
    st.title("Tabs")
    option = st.radio("Select an option", ["Introduction", "Domain Price Estimation", "Feature Explanation"])

# Display content based on the selected option
if option == "Introduction":

    # Set background color and padding
    st.markdown(
        """
        <style>
        body {
            background-color: #f0f2f6;
            padding: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Main title with welcome message
    st.title(":globe_with_meridians: Domain Name Price Estimation")
    st.subheader("Hello, hello! Welcome!")
    st.write("This app is used for predicting the price of a domain name based on its features.")
    st.write("Select a tab to get started.")

elif option == "Domain Price Estimation":
    # Input box for the domain name
    st.header("Enter a web domain name to get started:")
    domain = st.text_input("(e.g., google.com, dhjl.cn, snoopy.xyz, etc.):")

    # List of valid domain suffixes
    valid_suffixes = list(tld_scores.keys())

    if domain:
        # Check if the domain has a valid suffix
        suffix = '.' + domain.split('.')[-1]
        if suffix in valid_suffixes:
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"The entered domain name is: {domain}")
                st.markdown("---")
                st.write(f"Processing domain statistics for: {domain}, please wait...")

                output = calculate_factors(domain)

                # Option box for asking if the user wants to continue
                st.markdown("---")
                continue_option = st.selectbox("Do you want to continue to predict the price of the domain?", ["Select Option", "Yes", "No"])

            with col2:
                st.write("Here are the retrieved stats for that domain: ", output)
                st.markdown("---")

            if continue_option == "Yes":
                with col1:
                    st.write("Appending additional domain features...")
                    st.markdown("---")
                    extra_features(domain, output)
                    st.write("Calculating the price of the domain...")
                    temp = calculate_price(output, feature_weightages)
                    st.write(f"The calculated price of {domain} is {temp['Estimated Price'].values[0]:.2f} USD.")
                    st.markdown("---")

                with col2:
                    st.write("Here are the extra features for that domain: ", output)

            elif continue_option == "No":
                with col1:
                    # Code to handle if the user doesn't want to continue
                    st.write("Reenter another domain name if needed.")

        else:
            st.error(f"Error: Invalid domain suffix. Please enter a valid domain, e.g. google.com")
    else:
        st.write("Please enter a web domain name.")
else:
    st.title("Features Used to Calculate Domain Name Price")

    # Use columns layout manager to split text into two columns
    col1, col2 = st.columns(2)

    # First column
    with col1:
        st.markdown("<h3 style='color:lightblue;'>SEO Score</h3>", unsafe_allow_html=True)
        st.write("SEO Score is retrieved from Google’s PageSpeedOnline API to determine if the domain name has a good SEO score.")

        st.markdown("<h3 style='color:lightblue;'>Domain Age</h3>", unsafe_allow_html=True)
        st.write("Domain Age represents how long the domain name has been registered, measured in years. This information is retrieved from WHOIS databases.")

        st.markdown("<h3 style='color:lightblue;'>TLD Popularity Share</h3>", unsafe_allow_html=True)
        st.write("TLD Popularity Share is a scale from 0 to 100 based on the usage popularity of the top-level domain (TLD). For example, .com has a popularity share of 100, while newer or less common TLDs may have lower scores.")

        st.markdown("<h3 style='color:lightblue;'>Search Results Count</h3>", unsafe_allow_html=True)
        st.write("Search Results Count indicates the number of Google search results associated with the domain name. This information is retrieved from the Google Custom Search API.")

    # Second column
    with col2:
        st.markdown("<h3 style='color:lightblue;'>Name Composition</h3>", unsafe_allow_html=True)
        st.write("Name Composition calculates a score based on the meaningful words contained within the domain name. The score ranges from 0 to 100.")

        st.markdown("<h3 style='color:lightblue;'>Length Score</h3>", unsafe_allow_html=True)
        st.write("Length Score assigns a score based on the length of the domain name. For example, shorter domain names may receive a higher score than longer ones.")

        st.markdown("<h3 style='color:lightblue;'>Brand Score</h3>", unsafe_allow_html=True)
        st.write("Brand Score calculates a score based on keywords found in a Brand Name database, retrieved from TheCompaniesAPI. This helps determine if the domain name contains recognizable brand keywords.")
        
        st.markdown("<p style='color:red;'>Note: These are just some of the factors used in domain price estimation.</p>", unsafe_allow_html=True)



