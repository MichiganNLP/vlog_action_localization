from __future__ import print_function, absolute_import, unicode_literals, division

import pandas as pd
import glob
import os

def map_old_to_new_urls(path_old_urls, path_new_urls):

    list_old_csv_files = sorted(glob.glob(path_old_urls + "*.csv"))
    list_new_csv_files = sorted(glob.glob(path_new_urls + "*.csv"))

    for index in range(0, len(list_new_csv_files)):
        path_old_url = list_old_csv_files[index]
        path_new_url = list_new_csv_files[index]
        name_file = path_new_url.split("/")[-1]

        df_old = pd.read_csv(path_old_url)
        list_old_urls = list(df_old["Video_URL"])

        dict_new_urls = pd.read_csv(path_new_url, index_col=0, squeeze=True).to_dict()

        list_new_file = []
        for url_old in list_old_urls:
            if url_old in dict_new_urls.keys():
                list_new_file.append([url_old, dict_new_urls[url_old]])
            else:
                list_new_file.append([url_old, ''])
        df = pd.DataFrame(list_new_file)
        # df = df.transpose()
        df.columns = ["Video_URL", "Video_Name"]
        df.to_csv('data/new_video_urls/' + name_file, index=False)


def main():
    path_root = "/local/oignat/Action_Recog/large_data/"

    path_old_urls = path_root + "all_transcripts/video_urls/"
    path_new_urls = path_root + "youtube_data/video_urls/"
    map_old_to_new_urls(path_old_urls, path_new_urls)

if __name__ == '__main__':
    main()