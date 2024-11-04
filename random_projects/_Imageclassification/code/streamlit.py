import streamlit as st
#import person_classification as pc

#explanation of the application
st.header("The small Face Recognition Programm")

st.write("In this application allows you to recognize the following eight artists as if you would know them by yourself!! try it out!")

img = {"Kutcher": "../Dataset_images/ashton kutcher/15--er-ist-nicht-laenger-vorsitzender---1-1---spoton-article-1052486.jpg",
        "Cardi": "../Dataset_images/cardi b/00-promo-cardi-bty.jpg",
        "Rey": "../Dataset_images/lana del rey/70791_fancybox_1AqOkm_rNXvcG.jpg",
        "Brando": "../Dataset_images/marlon brando/363011_poster.jpg",
        "Williams": "../Dataset_images/robbie williams/29--so-erklaert-er-seinen-gewichtsverlust---3-2---spoton-article-1055009.jpg",
        "Sullivan": "../Dataset_images/ronnie o'sullivan/_133091343_gettyimages-2140892985.jpg",
        "Usher": "../Dataset_images/usher/STP-L-USHER-1104-02.jpg",
        "Weeknd": "../Dataset_images/weeknd/527f6551-d55c-4c97-be2b-90a38c88057e.jpg"}


# Homepage. Two columns
col1, col2, col3, col4 = st.columns(4, gap="medium")
with col1:
    st.image("../cropped_images/ashton kutcher/ashton kutcher5.png",width=150)
    st.write("Ashton Kutcher")
with col2:
    st.image("../cropped_images/cardi b/cardi b8.png",width=150)
    st.write("Cardi B")
with col3:
    st.image("../cropped_images/lana del rey/lana del rey10.png",width=150)
    st.write("Lana Del Rey")
with col4:
    st.image("../cropped_images/marlon brando/marlon brando6.png",width=150)
    st.write("Marlon Brando")

st.divider()
col1, col2, col3, col4 = st.columns(4, gap="medium")
with col1:
    st.image("../cropped_images/robbie williams/robbie williams3.png",width=150)
    st.write("Robbie Williams")
with col2:
    st.image("..//cropped_images/ronnie o'sullivan/ronnie o'sullivan9.png",width=150)
    st.write("Ronnie O'Sullivan")
with col3:
    st.image("../cropped_images/usher/usher2.png",width=150)
    st.write("Usher")
with col4:
    st.image("../cropped_images/weeknd/weeknd1.png",width=150)
    st.write("Weeknd")

st.divider()

#   Sidebar with Artist-selection and File-Uplooat
st.sidebar.header("Artists")

artists = {'Ashton Kutcher': "Kutcher",
    'Cardi B': "Cardi",
    'Lana Del Rey': "Rey",
    'Marlon Brando': "Brando",
    'Robbie Williams': "Williams",
    "Ronnie O'Sullivan": "Sullivan",
    'Usher': "Usher",
    'Weeknd': "Weeknd"}

select = st.sidebar.radio("for a brief biographie choose character : ", list(artists.keys()), index=None)
st.write(select)
if select:
    st.image(img[artists[select]], width=400)
    st.write("!!hier wird danach eine schnittstelle von Wikipedia eingebaut!!")


import os
import ml
import data_functions as df

# New Page: Run the model!
# drag and drop space.

file = st.sidebar.file_uploader(label="To start the machine learning process, upload the picture of an artist: ", type=["png", "jpg", "jpeg"])
if file is not None:
    st.image(file,width=200)
    
    with open(file.name, "wb") as f:
        f.write(file.getbuffer())
    st.success("File Saved!")
    
    file_details = file.name
    ml.lr.fit(ml.X_train, ml.y_train)


    prediction = ml.lr.predict(ml.guess_image(file_details))
    probability = ml.lr.predict_proba(ml.guess_image(file_details))
    """col1, col2 = st.columns(2)
    with col1:
        st.write(ml.lr.predict(ml.X_test))
    with col2:
        st.write(ml.y_test)"""
    st.write(ml.artist[prediction[0]])
    st.write(probability)




# ML process and 
# answer