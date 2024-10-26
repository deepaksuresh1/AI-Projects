from langchain_community.document_loaders import YoutubeLoader

from googleapiclient.discovery import build

import yaml
import os

import pandas as pd
import pytimetk as tk
from tqdm import tqdm


# 2.0 YOUTUBE API KEY SETUP 

PATH_CREDENTIALS = '../credentials.yml'

os.environ['YOUTUBE_API_KEY'] = yaml.safe_load(open(PATH_CREDENTIALS))['youtube'] 

# 3.0 VIDEO TRANSCRIPT SCRAPING FUNCTIONS

def search_videos(topic, api_key, max_results=20):
    youtube = build('youtube', 'v3', developerKey=api_key)
    request = youtube.search().list(
        q=topic,
        part='id,snippet',
        maxResults=max_results,
        type='video'
    )
    response = request.execute()
    video_ids = [item['id']['videoId'] for item in response['items']]
    return video_ids

def load_video(video_id):
    url = f'https://www.youtube.com/watch?v={video_id}'
    loader = YoutubeLoader.from_youtube_url(
        url, 
        add_video_info=True,
    )
    doc = loader.load()[0]
    doc_df = pd.DataFrame([doc.metadata])
    doc_df['video_url'] = url
    doc_df['page_content'] = doc.page_content
    return doc_df


# 4.0 SCRAPE YOUTUBE VIDEOS TRANSCRIPTS

TOPIC = "Social Media Brand Strategy Tips"

video_ids = search_videos(
    topic=TOPIC, 
    api_key=os.environ['YOUTUBE_API_KEY'], 
    max_results=50
)
video_ids

# * Scrape the video metadata and page content
videos = []
for video_id in tqdm(video_ids, desc="Processing videos"):
    try:
        video = load_video(video_id)
        videos.append(video)
    except Exception as e:
        print(f"Skipping video {video_id} due to error: {e}")


videos_df = pd.concat(videos, ignore_index=True)

# videos_df = pd.read_csv('data/youtube_videos.csv')

videos_df

videos_df.glimpse()

# * Store the video transcripts in a CSV File
videos_df.to_csv('data/youtube_videos.csv', index=False)