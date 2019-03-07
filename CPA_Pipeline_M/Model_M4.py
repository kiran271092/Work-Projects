
from itertools import chain
import numpy as np
import logging
import collections
import pandas as pd
import os

logging.basicConfig(level = logging.DEBUG)
logger = logging.getLogger()

class table_data:

    def change_page_y(self, Document, Merged_document, page_no):

        try:
            max_page = max(Document['Page'].unique().astype(int))
        except Exception as exe_error:
            logger.info("Unable to get page count")
            raise exe_error
        else:
            try:
                for i in range(page_no, max_page + 1):
                    max_y = max(Document[Document['Page'] == str(i)]['Y'])
                    Document.loc[Document['Page'] == str(i + 1), 'Y'] = Document.loc[Document['Page'] == str(
                        i + 1), 'Y'] + max_y + 1
                    Document.loc[Document['Page'] == str(i + 1), 'Y1'] = Document.loc[Document['Page'] == str(
                        i + 1), 'Y1'] + max_y + 1
                for i in range(page_no, max_page + 1):
                    max_y = max(Merged_document[Merged_document['Page'] == str(i)]['Y'])
                    Merged_document.loc[Merged_document['Page'] == str(i + 1), 'Y'] = Merged_document.loc[
                                                                                          Merged_document[
                                                                                              'Page'] == str(
                                                                                              i + 1), 'Y'] + max_y + 1
                    Merged_document.loc[Merged_document['Page'] == str(i + 1), 'Y1'] = Merged_document.loc[
                                                                                           Merged_document[
                                                                                               'Page'] == str(
                                                                                               i + 1), 'Y1'] + max_y + 1
                Document['Y'] = Document['Y'].astype(np.double)
                Merged_document['Y'] = Merged_document['Y'].astype(np.double)
                Document['Y2'] = Document['Y']
                Merged_document['Y2'] = Merged_document['Y']
                return Document, Merged_document
            except Exception as exe_error:
                logger.info("Unable to change page y")
                raise exe_error

    def get_flattened_headers(self, header_list):

        individual_list = []
        try:
            for header in header_list:
                individual_list.append(header.split())
            flatten_list = list(chain(*individual_list))
            try:
                assert len(flatten_list) > 0, "Unable to create flattened headers"
                return flatten_list
            except AssertionError as asserterror:
                logger.info(asserterror)
                raise asserterror
        except Exception as exe_error:
            raise exe_error

    def frequency_count(self, temp_Document, flatten_list):

        Document_filter = temp_Document[temp_Document['Word'].apply(lambda x: x in flatten_list)]
        Document_filter = Document_filter.groupby('Y').count().reset_index()
        Document_filter = Document_filter.sort_values(by='Y')
        return Document_filter

    def get_header_start_end_y(self, temp_Document_page, flattened_headers, start, end, headers):

        page_y_list = []
        y_min_final = []
        y_max_final = []
        y_words_final = []

        try:
            Document_filtered = self.frequency_count(temp_Document_page, flattened_headers)
            Document_filtered = Document_filtered.sort_values(by='Y')
            if len(Document_filtered) > 0:
                max_count_index = Document_filtered['Word'].idxmax()
                page_count = Document_filtered.loc[max_count_index, 'Word']
                page_y_list = list(Document_filtered[Document_filtered['Word'] == page_count]['Y'].unique())
                for i in range(0, len(page_y_list) - 1):
                    if page_y_list[i] >= 0 and page_y_list[i + 1] >= 0 and len(temp_Document_page[(temp_Document_page[
                                                                                                       'Y'] >
                                                                                                   page_y_list[i]) & (
                                                                                                          temp_Document_page[
                                                                                                              'Y'] <
                                                                                                          page_y_list[
                                                                                                              i + 1])]) <= 2:
                        page_y_list[i + 1] = -1
                page_y_list = [x for x in page_y_list if x >= 0]
            else:
                y_min_final.append(min(temp_Document_page['Y'].unique()))
                y_max_final.append(min(temp_Document_page['Y'].unique()))
            check_flattened_headers = flattened_headers.copy()
        except Exception as exe_error:
            logger.info("Unable to get headers")
            raise exe_error
        else:
            for each_y in page_y_list:
                y_min = []
                y_max = []
                for i in range(0, len(headers)):
                    temp_header = list(headers[i].split())
                    y1 = -1
                    y2 = -1
                    if len(temp_header) == 1:
                        check_doc = temp_Document_page[(temp_Document_page['Word'] == temp_header[0]) & (
                                    temp_Document_page['X_start'] >= start[i]) & (
                                                                   temp_Document_page['X_end'] <= end[i])]
                        y_list = list(check_doc['Y'].unique())
                        if len(y_list) == 1:
                            y1 = y_list[0]
                            y2 = y_list[0]
                        elif len(y_list) > 1:
                            diff = abs(each_y - y_list[0])
                            y1 = each_y
                            y2 = y_list[0]
                            for check_y in y_list:
                                if abs(each_y - check_y) <= diff:
                                    diff = abs(each_y - check_y)
                                    y1 = check_y
                                    y2 = check_y
                        else:
                            y1 = -2
                            y2 = -2
                        y_min.append(y1)
                        y_max.append(y2)
                    elif len(temp_header) > 0:
                        first_word = temp_header[0]
                        last_word = temp_header[len(temp_header) - 1]
                        check_doc_top = temp_Document_page[
                            (temp_Document_page['Word'] == first_word) & (temp_Document_page['X_start'] >= start[i]) & (
                                        temp_Document_page['X_end'] <= end[i])]
                        check_doc_bottom = temp_Document_page[
                            (temp_Document_page['Word'] == last_word) & (temp_Document_page['X_start'] >= start[i]) & (
                                        temp_Document_page['X_end'] <= end[i])]
                        y_list_top = list(check_doc_top['Y'].unique())
                        y_list_bottom = list(check_doc_bottom['Y'].unique())
                        if len(y_list_top) == 1:
                            y1 = y_list_top[0]
                        elif len(y_list_top) > 1:
                            diff = abs(each_y - y_list_top[0])
                            y1 = y_list_top[0]
                            for check_y in y_list_top:
                                if abs(each_y - check_y) <= diff:
                                    diff = abs(each_y - check_y)
                                    y1 = check_y
                        else:
                            y1 = -2
                        #print(first_word,check_doc_top['Word'],check_doc_top['X_start'],check_doc_top['X_end'])
                        #print(first_word, temp_Document_page[temp_Document_page['Word']==first_word]['Word'], temp_Document_page[temp_Document_page['Word']==first_word]['X_start'], temp_Document_page[temp_Document_page['Word']==first_word]['X_end'])
                        if len(y_list_bottom) == 1:
                            y2 = y_list_bottom[0]
                        elif len(y_list_bottom) > 1:
                            diff = abs(each_y - y_list_bottom[0])
                            y2 = y_list_bottom[0]
                            for check_y in y_list_bottom:
                                if abs(each_y - check_y) <= diff:
                                    diff = abs(each_y - check_y)
                                    y2 = check_y
                        else:
                            y2 = -2
                        #print(last_word,check_doc_bottom['Word'], check_doc_bottom['X_start'], check_doc_bottom['X_end'])
                    y_min.append(y1)
                    y_max.append(y2)
                    y_min = [x for x in y_min if x >= 0]
                    y_max = [x for x in y_max if x >= 0]
                if len(y_min) > 0 and len(y_max) > 0:
                    y_min_final.append(min(y_min))
                    y_max_final.append(max(y_max))
            try:
                # assert len(y_min_final) > 0, "No Y coordinate found here one"
                assert len(y_min_final) == len(y_max_final), "Number of Y coordinates do not match"
                return y_min_final, y_max_final
            except AssertionError as asserterror:
                logger.info(asserterror)
                raise asserterror

    def check_headers(self, temp_Document_page, flattened_headers, y_min_final, y_max_final, header_found):

        compare = lambda x, y: collections.Counter(x) == collections.Counter(y)
        y_min = []
        y_max = []
        for i in range(0, len(y_min_final)):
            check_headers = list(temp_Document_page[(temp_Document_page['Y'] >= y_min_final[i]) & (
                        temp_Document_page['Y'] <= y_max_final[i])]['Word'].values)
            if compare(flattened_headers, check_headers) is True:
                y_min.append(y_min_final[i])
                y_max.append(y_max_final[i])
                header_found = 1
            else:
                percent = 0
                if len(flattened_headers)>len(check_headers):
                    per_to_check = len(check_headers)/len(flattened_headers) * 100
                else:
                    per_to_check = len(flattened_headers)/len(check_headers) * 100
                for header in flattened_headers:
                    if header in check_headers:
                        check_headers.remove(header)
                        percent = percent + 1
                percent = (percent*per_to_check)/100
                if (percent / len(flattened_headers) * 100) > 90:
                    y_min.append(y_min_final[i])
                    y_max.append(y_max_final[i])
                    header_found = 1

        if len(y_min) == 0 and header_found>0:
            temp_y = list(temp_Document_page['Y'].unique())
            if len(temp_y) > 0:
                y_min.append(min(temp_y))
                y_max.append(min(temp_y))
                header_found = 1
        return y_min, y_max, header_found

    def get_page_y(self, temp_Document_page, flattened_headers, xstart, xend, headers, header_found):

        try:
            y_min_final, y_max_final = self.get_header_start_end_y(temp_Document_page, flattened_headers, xstart, xend,
                                                                   headers)
            y_min_final, y_max_final, header_found = self.check_headers(temp_Document_page, flattened_headers, y_min_final,
                                                          y_max_final,header_found)
            return y_min_final, y_max_final, header_found
        except Exception as exe_error:
            print("block")
            logger.info("Unable to get minimum and maximum Y for headers")
            raise exe_error

    def get_table_document(self, header_max_y, Merged_document):

        Document_table = Merged_document[(Merged_document['Y'] > header_max_y)]
        return Document_table

    def getting_columns(self, header_max_y, header_xstart, header_xend, header_list, Merged_document):

        x = 0
        columns_list = ["Char_Width", "Length", "Page", "Word", "X_end", "X_start", "Y", "word_id", "Heading"]
        df_table = pd.DataFrame(columns=columns_list)
        try:
            Document_table = self.get_table_document(header_max_y, Merged_document)
            if (len(header_list) > 1):
                for i in range(0, len(header_list)):
                    if (i == 0):
                        condition1 = header_xstart[i + 1]
                        if (header_xstart[i + 1] == -1):
                            condition1 = header_xend[i]
                        x = Document_table[(Document_table['X_end'] < condition1)][
                            'word_id']
                        y1 = Document_table[(Document_table['X_end'] < condition1)]
                        y1['Heading'] = header_list[i]
                        y1.reset_index(inplace=True)
                        df_table = df_table.append(y1)
                    elif (i == len(header_list) - 1):
                        condition2 = header_xend[i - 1]
                        if (header_xend[i - 1] == -1):
                            condition2 = header_xstart[i]
                        x = Document_table[(Document_table['X_start'] > condition2)]['word_id']
                        y1 = Document_table[(Document_table['X_start'] > condition2)]
                        y1['Heading'] = header_list[i]
                        y1.reset_index(inplace=True)
                        df_table = df_table.append(y1)
                    else:
                        condition1 = header_xend[i - 1]
                        if (header_xend[i - 1] == -1):
                            condition1 = header_xstart[i]
                        condition2 = header_xstart[i + 1]
                        if (header_xstart[i + 1] == -1):
                            condition2 = header_xend[i]
                        x = Document_table[(Document_table['X_end'] < condition2) & (
                                    Document_table['X_start'] > condition1)]['word_id']
                        y1 = Document_table[(Document_table['X_end'] < condition2) & (
                                    Document_table['X_start'] > condition1)]
                        y1['Heading'] = header_list[i]
                        y1.reset_index(inplace=True)
                        df_table = df_table.append(y1)
                    for i in x:
                        Document_table = Document_table[Document_table['word_id'] != i]
            return df_table
        except Exception as exe_error:
            logger.info("Unable to fetch tabule columns")
            raise exe_error

    def get_indexes(self, header_max_y, header_xstart, header_xend, header_list, df_merged):

        try:
            temp_df_table = self.getting_columns(header_max_y, header_xstart, header_xend, header_list, df_merged)
            df_table = temp_df_table.copy()
            df_table = df_table[df_table['Y'] > header_max_y]
            df_merged = df_merged[df_merged['Y'] > header_max_y]
            df_table_count = df_table.groupby('Y').count().reset_index()
            df_merged_count = df_merged.groupby('Y').count().reset_index()
            try:
                for index, cell in df_merged_count.iterrows():
                    if len(df_table_count[df_table_count['Y'] == cell['Y']]) < 1:
                        df_merged_count = df_merged_count[df_merged_count['Y'] != cell['Y']]
                y_list = df_table_count['Y'].values
                if len(df_table_count) > 0:
                    # max_count_index = max(df_table_count['Word'])
                    # max_count_word = df_table_count.loc[df_table_count['Word'].idxmax(), 'Word']
                    max_count_word = max(df_table.groupby('Y')['Word'].count())
                # print(y_list)
                for y in y_list:
                    # print(y)
                    df_table_word = df_table_count[df_table_count['Y'] == y]['Word'].values[0]
                    df_merged_word = df_merged_count[df_merged_count['Y'] == y]['Word'].values[0]
                    percent_captured = df_table_word / df_merged_word
                    percent_captured_max = df_table_word / max_count_word
                    #print(percent_captured,percent_captured_max)
                    if percent_captured < 0.8 or percent_captured_max < 0.5:
                        # print("Here")
                        df_table_count = df_table_count[df_table_count['Y'] != y]
                y_index = list(df_table_count['Y'].values)
                # y_index = list(y_index)
                y_index.insert(0, header_max_y)
                # print(y_index)
            except Exception as exe_error:
                logger.info("No rows")
                raise exe_error
            else:
                if (len(y_index) > 1):
                    in_between = []
                    for i in range(0, len(y_index)):
                        diff = abs(y_index[i] - y_index[i + 1])
                        if i == 0:
                            in_between.append(y_index[i + 1])
                        else:
                            in_between.append(y_index[i])
                            in_between.append(y_index[i + 1])
                        try:
                            gotdata = y_index[i + 2]
                        except IndexError:
                            gotdata = 'null'
                        if gotdata == 'null':
                            break
                        else:
                            if gotdata < (y_index[i + 1] + 2.5 * diff):
                                in_between.append(gotdata)
                            else:
                                break
                    y_index = list(set(in_between))
                y_index.sort()
            try:
                assert len(y_index) > 0, "No rows to fetch"
                return y_index, temp_df_table
            except AssertionError as asserterror:
                logger.info(asserterror)
                raise asserterror
        except Exception as exe_error:
            logger.info("Unable to fetch row")
            raise exe_error

    def merges_words(self, header_max_y, header_xstart, header_xend, header_list, df_merged):

        diff = 0
        try:
            table_indexes, df_table = self.get_indexes(header_max_y, header_xstart, header_xend, header_list, df_merged)
            if (len(table_indexes) > 0):
                if (len(table_indexes) == 1):
                    df_table = df_table[df_table['Y'] == table_indexes[0]]
                else:
                    for i in range(0, len(table_indexes)):
                        if (i < len(table_indexes) - 1):
                            diff = abs(table_indexes[i + 1] - table_indexes[i])
                        for header in header_list:
                            word = ""
                            counter = 0
                            wid = 0
                            wid_first = 0
                            if len(df_table[(df_table['Y'] >= table_indexes[i]) & (
                                    df_table['Y'] < table_indexes[i] + diff) & (df_table['Heading'] == header)][
                                       'Word']) > 1:
                                df_check = df_table[
                                    (df_table['Y'] >= table_indexes[i]) & (df_table['Y'] < table_indexes[i] + diff) & (
                                                df_table['Heading'] == header)]
                                for index, cell in df_check.iterrows():
                                    word = word + " " + cell['Word']
                                    wid = cell['word_id']
                                    if (counter == 0):
                                        wid_first = wid
                                    if (counter > 0):
                                        df_table = df_table[df_table['word_id'] != wid]
                                    counter += 1
                                df_table.loc[df_table['word_id'] == wid_first, 'Word'] = word
                    df_table = df_table[df_table['Y'] < max(table_indexes) + diff]
            return df_table
        except Exception as exe_error:
            logger.info("Unable to merge words")
            raise exe_error

    def get_table(self, header_max_y, header_xstart, header_xend, header_list, df_merged):

        try:
            df_table = self.merges_words(header_max_y, header_xstart, header_xend, header_list, df_merged)
            df = pd.DataFrame(columns=header_list)
            unique_y = list(df_table['Y'].unique())
            unique_y.sort()
            for each_y in unique_y:
                l = []
                for each_header in header_list:
                    x = df_table[(df_table['Heading'] == each_header) & (df_table['Y'] == each_y)]
                    if len(x) == 0:
                        l.append(np.NaN)
                    else:
                        m = df_table[(df_table['Heading'] == each_header) & (df_table['Y'] == each_y)]['Word']
                        l.append(m.values[0])
                row = pd.DataFrame([l], columns=header_list)
                df = df.append(row)
            return df
        except Exception as exe_error:
            logger.info("Unable to create dataframe")
            raise exe_error

    def get_final_header_list(self, input_path, temp_id):

        try:
            df = pd.read_csv(input_path)
        except Exception as exe_error:
            logger.info("Unable to find excel file")
            raise exe_error
        else:
            df = df[df['Id']==temp_id]
            words = list(df.loc[:,'Table_Headers'])
            X_start = list(df.loc[:,'Start'])
            X_end = list(df.loc[:,'End'])
            try:
                assert len(words) > 0, "No Headers"
                assert len(X_start) > 0, "No Headers"
                assert len(X_end) > 0, "No Headers"
                return X_start,X_end,words
            except AssertionError as asserterror:
                logger.error(asserterror)
                raise asserterror

    def extract_data(self, file_path, input_excel, Document, Merged_document, claim_list, temp_id):

        try:
            xstart, xend, headers = self.get_final_header_list(input_excel,temp_id)
        except Exception as exe_error:
            logger.info("Plaease provide correct path of excel file")
        else:
            try:
                output_file = os.path.splitext(file_path)[0]
                output_file = output_file+".xlsx"
                flattened_headers = self.get_flattened_headers(headers)
                name = os.path.basename(file_path).split('.')[0]
                claim_num = 1
                writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
                for i in range(0, len(claim_list)):
                    header_found = 0
                    if i == len(claim_list) - 1:
                        temp_Document = Document[Document['Page'].astype(int) >= claim_list[i]]
                        temp_Merged_document = Merged_document[Merged_document['Page'].astype(int) >= claim_list[i]]
                        start = claim_list[i]
                        end = -1
                    else:
                        temp_Document = Document[(Document['Page'].astype(int) >= claim_list[i]) & (
                                    Document['Page'].astype(int) < claim_list[i + 1])]
                        temp_Merged_document = Merged_document[
                            (Merged_document['Page'].astype(int) >= claim_list[i]) & (
                                        Merged_document['Page'].astype(int) < claim_list[i + 1])]
                        start = claim_list[i]
                        end = claim_list[i + 1]
                    num_pages = list(temp_Document['Page'].unique())
                    temp_Document, temp_Merged_document = self.change_page_y(temp_Document,
                                                                                     temp_Merged_document, start)
                    df = pd.DataFrame(columns=headers)
                    for each_page in num_pages:
                        temp_Document_page = temp_Document[temp_Document['Page'] == each_page]
                        temp_Merged_document_page = temp_Merged_document[temp_Merged_document['Page'] == each_page]
                        each_page_y_min, each_page_y_max, header_found = self.get_page_y(temp_Document_page,
                                                                                   flattened_headers, xstart, xend,
                                                                                   headers,header_found)
                        if len(each_page_y_min) == 1:
                            temp_document_table_page = self.get_table(each_page_y_max[0], xstart, xend, headers,
                                                                              temp_Merged_document_page)
                            df = df.append(temp_document_table_page)
                        elif len(each_page_y_min) > 1:
                            for i in range(0, len(each_page_y_min)):
                                if i == 0:
                                    temp_document_table_page_y = temp_Document_page[
                                        temp_Document_page['Y'] < each_page_y_min[i + 1]]
                                    temp_merged_document_table_page_y = temp_Merged_document_page[
                                        temp_Merged_document_page['Y'] < each_page_y_min[i + 1]]
                                elif i == len(each_page_y_min) - 1:
                                    temp_document_table_page_y = temp_Document_page[
                                        temp_Document_page['Y'] >= each_page_y_min[i]]
                                    temp_merged_document_table_page_y = temp_Merged_document_page[
                                        temp_Merged_document_page['Y'] >= each_page_y_min[i]]
                                else:
                                    temp_document_table_page_y = temp_Document_page[
                                        (temp_Document_page['Y'] >= each_page_y_min[i]) & (
                                                    temp_Document_page['Y'] < each_page_y_min[i + 1])]
                                    temp_merged_document_table_page_y = temp_Merged_document_page[
                                        (temp_Merged_document_page['Y'] >= each_page_y_min[i]) & (
                                                    temp_Merged_document_page['Y'] < each_page_y_min[i + 1])]

                                temp_document_table_page_y = self.get_table(each_page_y_max[i], xstart, xend,
                                                                                    headers,
                                                                                    temp_merged_document_table_page_y)
                                df = df.append(temp_document_table_page_y)
                    name = "claim_" + str(claim_num) + "_" + name + ".csv"
                    df.to_excel(writer, sheet_name='claim_'+str(claim_num),index=False)
                    claim_num = claim_num + 1
                writer.save()
                writer.close()
                logger.info("Check output " + output_file)
            except Exception as exe_error:
                logger.info("Unable to create output excel!!!")
