
# type: ignore
import streamlit as st
from st_clickable_images import clickable_images # pip install st_clickable_images
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Load MovieLens dataset (Ensure this dataset has an 'image_url' column)
@st.cache_data
def load_data():
    music = pd.read_csv("music.csv")
    interactions = pd.read_csv("interactions.csv")
    return music, interactions

music, interactions = load_data()

################################################# Popularity Recommendations Pre-Work ###########################################
pdata = music[['Title','Avg_rating','Rating_number']]
n = round(pdata['Rating_number'].mean(),2)
pdata['PopularityScore'] = round((pdata['Avg_rating']*pdata['Rating_number'])/n,2)

################################################# Content Recommendations Pre-Work ###########################################
cdata = music[['Music_id','Avg_rating','Rating_number','Price']]

# Cat Col Encoding
order = round(cdata.groupby('Music_id')['Rating_number'].sum(),2).sort_values().index
orders = {val:indx+1 for indx,val in enumerate(order)}
cdata.Music_id.replace(orders, inplace=True)


# Train NearestNeighbors model
cosinemodel = NearestNeighbors(n_neighbors=6, metric='cosine')   # 6 to include input item
cosinemodel.fit(cdata)

# Function to recommend music
def recommend_similar_items(itemtitle, n=5):
    # Finding the item index
    idx = music[music['Title']==itemtitle].index
    if len(idx) == 0:
        return []
    
    idx = idx[0]  # Take first match if multiple exist

    # Model input
    data = cdata.iloc[idx]
    
    # Find N nearest neighbors
    distances, indices = cosinemodel.kneighbors([data], n_neighbors=n+1)
    items_indices = indices.flatten()[1:]  # Skip input 
    
    return items_indices

################################################################# Streamlit UI #######################################################
st.subheader(":blue[🏷️ 🏷️ 🏷️ Digital Music Recommendations 🏷️ 🏷️ 🏷️]", divider=True)
st.subheader(":red[Popular Music..!]")

# Taking Top 10 based on Popularity Score
indxs = pdata.sort_values(by='PopularityScore', ascending=False)[0:10][['Title']].index

# Display images as clickable grid
image_paths = [eval(val)['large'] for val in music['Images'].iloc[indxs]]
captions = [val.title() for val in music['Title'].iloc[indxs]]
#features = [val.title() for val in items['Features'].iloc[indxs]]
ratings = [val for val in music['Avg_rating'].iloc[indxs]]
prices = [val for val in music['Price'].iloc[indxs]]

selected_index = clickable_images(
    image_paths,
    titles=captions,
    div_style={"display": "flex", "flex-wrap": "wrap", "gap": "30px"},
    img_style={"height": "90px", "border-radius": "10px", "cursor": "pointer"}
)

# Store selection
if selected_index is not None:
    st.session_state["selected_section"] = selected_index + 1

st.divider()

if "selected_section" in st.session_state:
    st.subheader(":green[Selected..]")
    indx = st.session_state['selected_section']-1
    col1, col2, col3 = st.columns(3)
    with col2:
        st.image(image_paths[indx], caption=captions[indx])
    #st.write(":green[Title:]")
    #st.write(features[indx])
    st.write(":green[Average_Rating:]", ratings[indx])
    st.write(":green[Price: $ ]", prices[indx])

    if st.button("Recommend Similar Items:"):
        st.divider()
        st.subheader(":red[Similar music..!]")
        title = captions[indx].lower()
        indxs = recommend_similar_items(title)

        # Display images as clickable grid
        image_paths = [eval(val)['large'] for val in music['Images'].iloc[indxs]]
        captions = [val.title() for val in music['Title'].iloc[indxs]]
        #features = [val.title() for val in items['Features'].iloc[indxs]]
        ratings = [val for val in music['Avg_rating'].iloc[indxs]]
        prices = [val for val in music['Price'].iloc[indxs]]

        selected_index2 = clickable_images(image_paths,titles=captions,
                                          div_style={"display": "flex", "flex-wrap": "wrap", "gap": "30px"},
                                          img_style={"height": "200px", "border-radius": "10px", "cursor": "pointer"}
                                          )

