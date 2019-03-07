
import logging
import collections
import fitz
import os
import re
import pandas as pd
from itertools import chain

logging.basicConfig(level = logging.DEBUG)
logger = logging.getLogger()

class M1_Heuristics:

    def Merge_line_wise(self,T_Document):
        Merged_Document = pd.DataFrame()
        row_temp = pd.DataFrame()
        for index, row_m in T_Document.iterrows():
            if re.match('^[|_]+$', row_m['Word']) is not None:
                Merged_Document = pd.concat([Merged_Document, pd.DataFrame(row_temp).T], axis=0)
                continue
            else:
                if (len(row_temp) == 0):
                    row_temp = row_m
                else:
                    if (row_m['X_start'] < (row_temp['X_end'] + 2.9 * row_temp['Char_Width'])) & (
                            row_m['Y1'] == row_temp['Y1']) & (
                            row_m['Y2'] == row_temp['Y2']) & (row_m['Page'] == row_temp['Page']) & (
                            row_temp['Word'][-1] != ':'):
                        row_temp['Word'] = (row_temp["Word"] + " " + row_m['Word'])
                        row_temp['Y2'] = row_m['Y2']
                        row_temp['Y1'] = row_m['Y1']
                        row_temp['Page'] = row_m['Page']
                        row_temp['Char_Width'] = (row_temp['Char_Width'] + row_m['Char_Width'])/2
                        row_temp['X_end'] = row_m['X_end']
                        # row_temp['Length'] = (row_temp['Length']+row_m['Length']+1)
                    else:
                        Merged_Document = pd.concat([Merged_Document, pd.DataFrame(row_temp).T], axis=0)
                        row_temp = row_m
        Merged_Document = pd.concat([Merged_Document, pd.DataFrame(row_temp).T], axis=0)
        return Merged_Document

    def heuristic_m1(self,filepath,doc):
        # doc = get_Document_Corpus(
        #     r'C:\Users\kiran.kandula\Downloads\Final_pipeline(1)\Final_pipeline\ca.creditservicerb.20180703113219.522.KC977411_20180703_104520838.pdf')

        final_df = pd.DataFrame()

        final_dict = {}
        for i in doc['Page'].astype(int).unique().tolist():
            df = doc[doc['Page'] == str(i)]
            df['Page'] = df['Page'].astype(int)
            final_df = pd.DataFrame()
            final_df = final_df.append(df)
            final_df.sort_values(by=['Y1', 'X_start'], inplace=True)

            merged_data = self.Merge_line_wise(final_df)
            regex_df = merged_data[merged_data['Word'].str[-1:].str.contains(":")]
            dict = {}
            for j in regex_df['Word'].values.tolist():
                dict[j] = regex_df[regex_df['Word'] == j][['X_start', 'Y1', 'X_end', 'Y2']].values.T[:, 0].tolist()
            final_dict[i] = dict
        return final_dict

    def merging_with_spacy_output(self, heuristic, spacy_output):

        heuristic_df = pd.concat({i: pd.DataFrame(j).T for i, j in heuristic.items()}, axis=0)
        heuristic_df = heuristic_df.reset_index()
        heuristic_df_dup = heuristic_df.drop_duplicates('level_1')
        #applying regex rules
        heuristic_df_dup['level_1'] = heuristic_df_dup['level_1'].str.replace(r"\*(.*?)\*", '')
        heuristic_df_dup = heuristic_df_dup[~heuristic_df_dup['level_1'].str.match('[0-9]+')]

        spacy_output_keys_list = list(spacy_output.keys())
        content_keys = []
        for key in spacy_output_keys_list:
            temp_keys = list(spacy_output[key].keys())
            for x in temp_keys:
                content_keys.append(x)
        heuristic_df_add = heuristic_df_dup[heuristic_df_dup['level_1'].apply(lambda x: x in list(set(heuristic_df_dup['level_1']).intersection(content_keys)))]
        df = pd.DataFrame(columns=heuristic_df_add.columns)
        for key in spacy_output_keys_list:
            content_keys = list(spacy_output[key].keys())
            for content in content_keys:
                temp_df = pd.DataFrame(columns=heuristic_df_add.columns)
                temp_df.loc[0, 'level_0'] = key
                temp_df.loc[0, 'level_1'] = content
                temp_df.loc[0, 0] = spacy_output[key][content][0]
                temp_df.loc[0, 1] = spacy_output[key][content][1]
                temp_df.loc[0, 2] = spacy_output[key][content][2]
                temp_df.loc[0, 3] = spacy_output[key][content][3]
                df = df.append(temp_df)
        df = df.append(heuristic_df_add)
        df_key_list = list(df['level_0'].unique())
        page_output = {}
        for key in df_key_list:
            temp_df = df[df['level_0'] == key]
            content = {}
            for index, cell in temp_df.iterrows():
                pos = []
                pos.append(cell[0])
                pos.append(cell[1])
                pos.append(cell[2])
                pos.append(cell[3])
                content[cell['level_1']] = pos
            page_output[key] = content
        #print(df_key_list)
        #heuristic_df_add = heuristic_df_add.drop('level_1', axis=1)
        # setting the level and transpose to get the columns as labels and converting it to dict by inferencing as list

        # heuristic_df_add = heuristic_df_add.set_index('level_0').T.to_dict('list')
        # spacy_output.update(heuristic_df_add)
        return page_output

    def reverse_spacy(self,Document,spacy_output):

        output = {}
        no_of_pages = list(spacy_output.keys())
        page_data = {}
        for page_no in no_of_pages:
            Document2 = Document[Document['Page'] == str(page_no)]
            Document2 = Document2.reset_index()
            Document2.drop('index', axis=1, inplace=True)
            Document2.sort_values(by=['Y_round', 'X_start'], ascending=True, inplace=True)
            Document2 = Document2.reset_index()
            Document2.drop('index', axis=1, inplace=True)
            df = pd.DataFrame()

            df = df.append(Document2.loc[Document2.index[0], Document2.columns.tolist()])
            for i in range(1, len(Document2) - 1):
                df = df.append(Document2.loc[i, Document2.columns.tolist()])
                df = df.reset_index(drop=True)

                if (df.loc[len(df) - 2, 'Y_round'] != df.loc[len(df) - 1, 'Y_round']):
                    df.set_value(len(df), ['Word', 'Y_round'], ["\\n", df.loc[len(df) - 1, 'Y_round']])
                    b, c = df.iloc[len(df) - 2].copy(), df.iloc[len(df) - 1].copy()
                    df.iloc[len(df) - 2], df.iloc[len(df) - 1] = c, b

            Document2 = df.copy()
            Document2['Length'] = Document2['Word'].apply(lambda x: len(x))
            prev_length = -1
            length_till_iterator = []
            for index, row in Document2.iterrows():
                length_till_iterator.append(prev_length + 1 + row['Length'])
                if (row['Word'] != "\\n"):
                    prev_length = prev_length + 1 + row['Length']
                else:
                    prev_length = prev_length + row['Length']
            Document2['Length_till_iterator'] = length_till_iterator
            spacy_cordinate_start = []
            spacy_cordinate_end = []
            prev_length = -1
            for index, row in Document2.iterrows():
                spacy_cordinate_start.append(prev_length + 1)
                if (index == 0):

                    spacy_cordinate_end.append(spacy_cordinate_start[index] + row['Length'] - 1)
                else:
                    spacy_cordinate_end.append(spacy_cordinate_start[index] + row['Length'])
                if (row['Word'] != "\\n"):
                    prev_length = row['Length_till_iterator']
                else:
                    prev_length = row['Length_till_iterator'] - 1

            Document2['spacy_cordinate_start'] = spacy_cordinate_start
            Document2['spacy_cordinate_end'] = spacy_cordinate_end

            for x in spacy_output[page_no].keys():
                output = {}
                for label in spacy_output[page_no][x].keys():
                    start = spacy_output[page_no][x][label][0]
                    end = spacy_output[page_no][x][label][1]
                    the_label_parts = pd.DataFrame()
                    for part in label.split():
                        filter_data = Document2[
                            (Document2['Word'].str.contains(part)) & (abs(Document2['Length'] - len(part)) <= 2)]
                        top = filter_data[
                            (filter_data['spacy_cordinate_start'] >= start) & (
                                        filter_data['spacy_cordinate_end'] <= end)]
                        the_label_parts = pd.concat([the_label_parts, top], axis=0)
                    the_label_parts = the_label_parts.sort_values(by=['Y_round', 'X_start'], ascending=True)
                    output[label] = [the_label_parts['X_start'].min(), the_label_parts['Y1'].min(),
                                     the_label_parts['X_end'].max(), the_label_parts['Y2'].max()]
                page_data[page_no] = output
        return page_data

    def highlight_headers(self,claim_document_path,output):

        doc = fitz.open(claim_document_path)
        key_list = list(output.keys())
        for each_key in key_list:
            content_keys = list(output[each_key].keys())
            page = doc[each_key - 1]
            for each_content in content_keys:
                list_of_coordinates = list(output[each_key][each_content])
                r = fitz.Rect(list_of_coordinates[0], list_of_coordinates[1], list_of_coordinates[2],
                              list_of_coordinates[3])
                page.addHighlightAnnot(r)
        name = os.path.split(claim_document_path)[1]
        path_pdf = os.path.split(claim_document_path)[0]
        name = "M1_" + name
        name = os.path.join(path_pdf, name)
        doc.save(name, garbage=4, deflate=True, clean=True)
        doc.close()


class M2_Heuristics:
    
    def get_flattened_headers(self,header_list):
        
        individual_list=[]
        try:
            for header in header_list:
                individual_list.append(header.split())
            flatten_list = list(chain(*individual_list))
            assert len(flatten_list)>0,"No headers for table found"
            return flatten_list
        except AssertionError as asserterror:
            logger.info(asserterror)
            raise asserterror
        except Exception as exe_error:
            logger.info("Unable to flatten headers")
            raise exe_error
    
    def get_headers_for_dict(self,header_list):
        try:
            flatten_list = list(chain(*header_list))
            assert len(flatten_list)>0,"No headers for table found"
            return flatten_list
        except AssertionError as asserterror:
            logger.info(asserterror)
            raise asserterror
        except Exception as exe_error:
            logger.info("Unable to flatten headers")
            raise exe_error

    def frequency_count(self,Document,header_list):
        try:
            flatten_list = self.get_flattened_headers(header_list)
            #Document,Merged_document = self.change_page_y(Document,Merged_document,start,logger)
        except Exception as exe_error:
            raise exe_error
        else:
            Document_filter = Document[Document['Word'].apply(lambda x:x in flatten_list)]
            Document_filter = Document_filter.groupby('Y').count().reset_index()
            Document_filter = Document_filter.sort_values(by = 'Y')
            try:
                assert len(Document_filter)>0,"Filtered Dataframe is empty"
                return Document_filter
            except AssertionError as asserterror:
                logger.info(asserterror)
                raise asserterror
    
    def get_header_list(self,Document,Merged_document,header_list):
        
        header_word_list = []
        y_list = []
        try:
            Document_filtered = self.frequency_count(Document,header_list)
        except Exception as error:
            raise error
        else:
            try:
                Document_filtered = Document_filtered.sort_values(by='Y')
                max_count_index = Document_filtered['Word'].idxmax()
                y_max = Document_filtered.loc[max_count_index,'Y']
            except Exception as exe_error:
                logger.info("Unable to find maximum y")
                raise exe_error
            else:
                header_word_list.append(list(Document[Document['Y']==y_max]['Word'].values))
                y_list.append(y_max)
                if max_count_index > 0:
                    y_above = Document_filtered.loc[max_count_index-1,'Y']
                    check_df = Merged_document[(Merged_document['Y']<y_max) & (Merged_document['Y']>y_above)]
                    if len(check_df)==0 and abs(y_max-y_above)<25:
                        header_word_list.append(list(Document[Document['Y']==y_above]['Word'].values))
                        y_list.append(y_above)
                if max_count_index < len(Document_filtered)-1:
                    y_below = Document_filtered.loc[max_count_index+1,'Y']
                    check_df = Merged_document[(Merged_document['Y']>y_max) & (Merged_document['Y']<y_below)]
                    if len(check_df)==0 and abs(y_max-y_below)<25:
                        header_word_list.append(list(Document[Document['Y']==y_below]['Word'].values))
                        y_list.append(y_below)
                try:
                    assert len(header_word_list)>0,"Header list is empty"
                    assert len(y_list)>0,"Y coordinate list is empty"
                    return header_word_list,y_list
                except AssertionError as asserterror:
                    logger.info(asserterror)
                    raise asserterror
                    
    def match_header(self,header1,header2):
    
        compare = lambda x, y: collections.Counter(x) == collections.Counter(y)
        if compare(header1,header2) is True:
            return True
        else:
            return False

    def get_headers(self,Document,Merged_Document,header,Y):

        final_header = {}
        final_y = {}
        key_list = list(header.keys())
        if len(key_list)<1:
            return final_header,final_y
        elif len(key_list)==1:
            return header,Y
        else:
            try:
                final_header[key_list[0]] = header[key_list[0]]
                final_y[key_list[0]] = Y[key_list[0]]
                for key in range (1,len(key_list)):
                    check = True
                    temp_header = header[key_list[key]]
                    temp_y = Y[key_list[key]]
                    check = self.match_header(final_header[key_list[0]],temp_header)
                    if check==False:
                        final_header[key_list[key]] = temp_header
                        final_y[key_list[key]] = temp_y
            except Exception as exe_error:
                logger.info("Unable to retrieve headers")
                raise exe_error
            else:
                try:
                    assert len(final_header)>0,"No header found"
                    assert len(final_y)>0,"Unable to get Y coordinate"
                    return final_header,final_y
                except AssertionError as asserterror:
                    logger.info(asserterror)
                    raise asserterror

    def set_values(self):
        return -12345,-12345,-12345,-12345,-12345
    
    def merge_multiline_headers(self,Document,Merged_document,y_dict):
        
        words = []
        xstart = []
        xend = []
        y1 = []
        y2 = []
        final_coordinates = {}
        try:
            key_list = list(y_dict.keys())
            for key in key_list:
                temp_dict = {}
                y_list = y_dict[key]
                Multiple_lines = Merged_document[(Merged_document['Page']==str(key)) & (Merged_document['Y'].apply(lambda x:x in y_list))].sort_values(by=["X_start","Y"])
                words = list(Multiple_lines['Word'].values)
                xstart = list(Multiple_lines['X_start'].values)
                xend = list(Multiple_lines['X_end'].values)
                y1 = list(Multiple_lines['Y1'].values)
                y2 = list(Multiple_lines['Y2'].values)
                highlight_list = []
                for i in range(0,len(words)):
                    count = 0
                    if i<len(words)-2:
                        if words[i]!=-12345 and words[i+1]!=-12345 and words[i+2]!=-12345:
                            if y2[i]!=y2[i+1] and y2[i+1]!=y2[i+2]:
                                if y2[i]==y2[i+2]:
                                    if xend[i]>xstart[i+1] and xend[i+1]>xstart[i+2]:
                                        words[i+1],xstart[i+1],xend[i+1],y2[i+1],y1[i+1] = self.set_values()
                                        highlight_list.append(words[i])
                                        count = 1
                            elif y2[i]!=y2[i+1] and y2[i]!=y2[i+2]:
                                if y2[i+1]==y2[i+2]:
                                    if xend[i]>xstart[i+1] and xend[i]>xstart[i+2]:
                                        words[i],xstart[i],xend[i],y2[i],y1[i] = self.set_values()
                                        count = 1
                    if count==0 and i<len(words)-1:
                        first = range(int(xstart[i]),int(xend[i]+1))
                        second = range(int(xstart[i+1]),int(xend[i+1]+1))
                        first_set = set(first)
                        result = bool(first_set.intersection(second))
                        if result is True:
                            if(y2[i]<y2[i+1]):
                                temp_word = words[i]+" "+words[i+1]
                                words[i+1] = temp_word
                                if(xstart[i]<xstart[i+1]):
                                    xstart[i+1] = xstart[i]
                                if(xend[i]>xend[i+1]):
                                    xend[i+1] = xend[i]
                                y1[i+1] = y1[i]
                                words[i],xstart[i],xend[i],y2[i],y1[i] = self.set_values()
                            else:
                                temp_word = words[i+1]+" "+words[i]
                                words[i+1] = temp_word
                                if(xstart[i]<xstart[i+1]):
                                    xstart[i+1] = xstart[i]
                                if(xend[i]>xend[i+1]):
                                    xend[i+1] = xend[i]
                                y2[i+1] = y2[i]
                                words[i],xstart[i],xend[i],y2[i],y1[i] = self.set_values()
                words = [x for x in words if x!=-12345]
                xstart = [x for x in xstart if x!=-12345]
                xend = [x for x in xend if x!=-12345]
                y1 = [x for x in y1 if x!=-12345]
                y2 = [x for x in y2 if x!=-12345]
                temp_dict['X_start'] = xstart
                temp_dict['X_end'] = xend
                temp_dict['Y1'] = y1
                temp_dict['Y2'] = y2
                temp_dict['Headers'] = words
                final_coordinates[key] = temp_dict 
            temp_dict = {}
            for key in key_list:
                if len(final_coordinates[key]['X_start'])==len(final_coordinates[key]['X_end']) and len(final_coordinates[key]['X_end'])==len(final_coordinates[key]['Y1']) and len(final_coordinates[key]['Y1'])==len(final_coordinates[key]['Y2']):
                    temp_dict[key] = final_coordinates[key]
            final_coordinates = temp_dict
            try:
                key_list = list(y_dict.keys())
                for key in key_list:
                    assert len(final_coordinates[key]['X_start'])==len(final_coordinates[key]['X_end']),"Number of x and y coordinates do no match in page"+str(key)
                    assert len(final_coordinates[key]['X_end'])==len(final_coordinates[key]['Y1']),"Number of x and y coordinates do no match in page"+str(key)
                    assert len(final_coordinates[key]['Y1'])==len(final_coordinates[key]['Y2']),"Number of x and y coordinates do no match in page"+str(key)
                    return final_coordinates
            except AssertionError as asserterror:
                logger.info(asserterror)
                raise asserterror
        except Exception as exe_error:
            logger.info("Unable to fetch final coordinates")
            raise exe_error
    
    def highlight_headers(self,file_path,final_dict):

        try:
            doc = fitz.open(file_path)
            page_count = doc.pageCount
            key_list = list(final_dict.keys())
            for key in key_list:
                if key<=page_count:
                    page = doc[int(key)-1]
                    x1 = final_dict[key]['X_start']
                    x2 = final_dict[key]['X_end']
                    y1 = final_dict[key]['Y1']
                    y2 = final_dict[key]['Y2']
                    for i in range(0,len(x1)):
                        if x2[i]-x1[i]>=2 :
                            r = fitz.Rect(x1[i],y1[i],x2[i],y2[i])
                            highlight = page.addHighlightAnnot(r)
            name = os.path.split(file_path)[1]
            path_pdf = os.path.split(file_path)[0]
            name = "M2_" + name
            name = os.path.join(path_pdf,name)
            doc.save(name, garbage=4, deflate=True, clean=True)
        except Exception as exe_error:
            logger.info("Unable to find pdf")
            raise exe_error
        finally:
            doc.close()


class heuristics:

    def m1(self,file_path,header_list,Document):
        document_tasks_object = M1_Heuristics()
        try:
            spacy_pixel_cooordinates = document_tasks_object.reverse_spacy(Document,header_list)
            heuristics_coordinates = document_tasks_object.heuristic_m1(file_path,Document)
            spacy_pixel_cooordinates1 = document_tasks_object.merging_with_spacy_output(heuristics_coordinates,spacy_pixel_cooordinates)
            print(spacy_pixel_cooordinates)
            print(spacy_pixel_cooordinates1)
            document_tasks_object.highlight_headers(file_path,spacy_pixel_cooordinates1)
        except:
            doc = fitz.open(file_path)
            name = os.path.split(file_path)[1]
            path_pdf = os.path.split(file_path)[0]
            name = "M1_" + name
            name = os.path.join(path_pdf, name)
            doc.save(name, garbage=4, deflate=True, clean=True)
            doc.close()


    def m2(self,file_path,header_list,Document,Merged_Document):
        document_tasks_object = M2_Heuristics()
        try:
            key_list = list(header_list.keys())
            header_word_dict = {}
            y_dict = {}
            count = 0
            is_predicted = False
            for key in key_list:
                if len(header_list[key])>0 and count<=2:
                    page_header = header_list[key]
                    page_document = Document[Document['Page']==str(key)]
                    page_merged_document = Merged_Document[Merged_Document['Page']==str(key)]
                    header_word,y_dict[key] = document_tasks_object.get_header_list(page_document,page_merged_document,page_header)
                    header_word_dict[key] = document_tasks_object.get_headers_for_dict(header_word)
                    is_predicted = True
                count = count+1
            if is_predicted == True:
                header_word_dict,y_dict = document_tasks_object.get_headers(Document,Merged_Document,header_word_dict,y_dict)
                final_dict = document_tasks_object.merge_multiline_headers(Document,Merged_Document,y_dict)
                document_tasks_object.highlight_headers(file_path,final_dict)
            else:
                doc = fitz.open(file_path)
                name = os.path.split(file_path)[1]
                path_pdf = os.path.split(file_path)[0]
                name = "M2_" + name
                name = os.path.join(path_pdf, name)
                doc.save(name, garbage=4, deflate=True, clean=True)
            logger.info("Pdf highlighted,please check it!!!")

        except Exception as exe_error:
            logger.info("Unable to highlight pdf,please manually highlight it!!!")

