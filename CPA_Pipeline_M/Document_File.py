
import pandas as pd
import json
import subprocess
import numpy as np
import re
import warnings
import logging
import argparse
import ast

import Spacy_Model
import PDF_Highlight
import Saving_In_Excel
import Model_M4
import Model_M3


warnings.filterwarnings('ignore')
logging.basicConfig(level = logging.DEBUG)
logger = logging.getLogger()


class create_document:
    
    def getCordinatesForDocument(self,x):
        try:
            os_object = subprocess.run('java -cp last_CPA_with_width.jar;v/*;. GetCharLocationAndSize "%s" ' %(x),stdout=subprocess.PIPE, encoding='ascii')
        except UnicodeDecodeError:
            os_object = subprocess.run('java -cp last_CPA_with_width.jar;v/*;. GetCharLocationAndSize "%s" ' %(x),stdout=subprocess.PIPE, encoding='latin-1')
        except:
            os_object = subprocess.run('java -cp last_CPA_with_width.jar;v/*;. GetCharLocationAndSize "%s" ' %(x),stdout=subprocess.PIPE, encoding='utf-8')   
        words_cordinates = os_object.stdout
        return words_cordinates

    def get_Document_Corpus(self,file_path):
        all_co_ordinates=self.getCordinatesForDocument(file_path)
        all_co_ordinates_seperated=all_co_ordinates.split('\n')
        json_object_list=[]
        for x in all_co_ordinates_seperated:
            if x.strip()!='':
                json_object_list.append(json.loads(x.strip()))
        Document=pd.DataFrame(columns=['Word','X_start','X_end','X_center','Y','Page','Char_Width','Length','new_width','new_height'])
        Merged_Document=pd.DataFrame(columns=['Word','X_start','X_end','X_center','Y','Page','Char_Width','Length','new_width','new_height'])
        char_width_t2 = 1
        for m in json_object_list:
            bet = " "
            word_t = ""
            for i in m:
                if re.match('^[|_]+$', i['word']) is not None:
                    bet = ""
                else:
                    if len(i['word'])>0:
                        word=i['word']
                        length=len(word)
                        page=i['page']
                        x_start = i['cordinates']['X1']
                        x_end = i['cordinates']['X'+str(length)]+i['cordinates']['W'+str(length)]
                        if length == 1:
                            char_width = char_width_t2                       
                        else:
                            char_width = (x_end - x_start)/(length-1)
                            char_width_t2 = max(1,char_width)
                        if length%2==0:
                            x_center=(i['cordinates']['X'+str(int(length/2))]+i['cordinates']['X'+str(int(length/2)+1)])/2
                        else:
                            x_center=i['cordinates']['X'+str(int(length/2)+1)]
                        a=0
                        for z in [ i for i in i['cordinates'].keys() if 'Y' in i]:
                            a+=i['cordinates'][z]
                        yy = round(a/length)
                        h_list = []
                        for z in [ i for i in i['cordinates'].keys() if 'H' in i]:
                            h_list.append(i['cordinates'][z])
                        hh = max(h_list)
                        w_list =[]
                        for z in [i for i in i['cordinates'].keys() if 'W' in i]:
                            w_list.append(i['cordinates'][z])
                        ww = max(w_list)
                        row=pd.DataFrame(data=[[word,x_start,x_end,x_center,yy,page,char_width,length,ww,hh]],columns=['Word','X_start','X_end','X_center','Y','Page', 'Char_Width','Length','new_width','new_height'])
                        Document=pd.concat([Document,row],axis=0) 
                        Document.reset_index(inplace=True,drop=True)
                        Document['word_id']=range(0,(len(Document)))
                        Document['Y1']=Document['Y']-Document['new_height']
                        Document['Y2']=Document['Y']
                        Document['Y_avg']=((Document['Y1']+Document['Y2'])/2)
                        Document['Y_round']=np.around(Document['Y'].astype(np.double),2)
                        Document['X_start']=np.around(Document['X_start'].astype(np.double),2)
                        Document['X_end'] = np.around(Document['X_end'].astype(np.double), 2)
                        if word_t == "":
                            word_t = word
                            x_start_t = x_start
                            x_end_t = x_end
                            x_center_t = x_center
                            y_t = yy
                            page_t = page
                            char_width_t = char_width
                            char_width_t2 = max(1,char_width)
                            ww_t = ww
                            hh_t = hh
                        else:
                            if (x_start < (x_end_t + 2.8*char_width_t)) & (y_t == yy) & (word_t[-1] != ':'):                            
                                word_t += bet + word
                                x_end_t = x_end
                                x_center_t = (x_end_t + x_start_t)/2
                                if len(word_t) == 1:
                                    char_width_t = char_width_t2
                                else:
                                    char_width_t = (x_end_t - x_start_t)/(len(word_t)-1)
                                    char_width_t2 = max(char_width_t,1)
                            else:
                                if len(word_t)>0:
                                    if word_t[-1] == ':':
                                        word_t = word_t[:-1]
                                    row=pd.DataFrame(data=[[word_t,x_start_t,x_end_t,x_center_t,y_t,page_t,char_width_t,len(word_t),ww_t,hh_t]],columns=['Word','X_start','X_end','X_center', 'Y','Page','Char_Width','Length','new_width','new_height'])
                                    Merged_Document=pd.concat([Merged_Document,row],axis=0)
                                    Merged_Document.reset_index(inplace=True,drop=True)
                                    Merged_Document['word_id']=range(0,(len(Merged_Document)))
                                    word_t = word
                                    x_start_t = x_start
                                    x_end_t = x_end
                                    x_center_t = x_center
                                    y_t = yy
                                    page_t = page
                                    char_width_t = char_width
                                    char_width_t2 = max(1,char_width_t)
                    bet = " "
            if len(word_t)>0:
                row=pd.DataFrame(data=[[word_t,x_start_t,x_end_t,x_center_t, y_t,page_t,char_width_t,len(word_t),ww_t,hh_t]],columns=['Word','X_start','X_end','X_center','Y','Page','Char_Width','Length','new_width','new_height'])
                Merged_Document=pd.concat([Merged_Document,row],axis=0)
                Merged_Document.reset_index(inplace=True,drop=True)
                Merged_Document['word_id']=range(0,(len(Merged_Document)))
                Merged_Document['Y1']=Merged_Document['Y']-Merged_Document['new_height']
                Merged_Document['Y2']=Merged_Document['Y']
                Merged_Document['Y_avg']=((Merged_Document['Y1']+Merged_Document['Y2'])/2)
                Merged_Document['Y_round']=np.round(Merged_Document['Y'].astype(np.double),2)
                Merged_Document['X_start'] = np.around(Merged_Document['X_start'].astype(np.double), 2)
                Merged_Document['X_end'] = np.around(Merged_Document['X_end'].astype(np.double), 2)
        try:
            assert len(Document)>0,"Empty DataFrame of words,may be file path is not specified properly or it is completely image"
            assert len(Merged_Document)>0,"Empty Merged DataFrame of words,may be file path is not specified properly"
            return Document, Merged_Document
        except AssertionError as asserterror:
            logger.info(asserterror)

    def get_spacy_json_from_Document(self,Document,claim_list):

        if len(claim_list)==0:
            return -1
        else:
            if len(claim_list)>=2:
                start = claim_list[0]
                end = claim_list[1]
            else:
                start = claim_list[0]
                end = int(np.max(Document['Page']))+1
            final_annotated_text={}
            no_of_pages=int(np.max(Document['Page']))
            for claim_page in range(start,end):
                page_annotated_text={}
                Document2 = Document[Document['Page']==str(claim_page)]
                Document2.sort_values(by=['Y','X_start'],ascending=True,inplace=True)
                Document2['Y_round']=np.around(Document2['Y'].astype(np.double),2)
                data=pd.DataFrame(columns=Document2.columns)
                old_y=Document2['Y_round'].min()
                for index,row in Document2.iterrows():
                    if row['Y_round'] != old_y:
                        data_n = pd.DataFrame(columns=Document2.columns)
                        data_n.loc[0,'Word'] = "\n"
                        data_n.loc[0,'X_start'] = 99
                        data_n.loc[0,'X_end'] = 99
                        data_n.loc[0,'X_center'] = 99
                        data_n.loc[0,'Y'] = old_y
                        data_n.loc[0,'Page'] = '1'
                        data_n.loc[0,'Char_Width'] = 2
                        data_n.loc[0,'Length'] = 2
                        data_n.loc[0,'new_width'] = 2
                        data_n.loc[0,'new_height'] = 2
                        data_n.loc[0,'word_id'] = -1
                        data_n.loc[0,'Y1'] = old_y
                        data_n.loc[0,'Y2'] = old_y
                        data_n.loc[0,'Y_round'] = old_y
                        old_y = row['Y']
                        data = data.append(data_n)
                    data = data.append(row)
                Document2=data
                Document2['Length']=Document2['Word'].apply(lambda x: len(x))
                content=' '.join(Document2['Word'])
                page_annotated_text['content']=content
                final_annotated_text[claim_page] = page_annotated_text
            return final_annotated_text

if __name__ == '__main__':
    #
    # ap = argparse.ArgumentParser()
    # ap.add_argument("--file_path=", required=True,
    #                 help="path of pdf")
    # ap.add_argument("--claim_list=", required=True,
    #                 help="list of claim page in pdf")
    # ap.add_argument("--template_id=", required=True,
    #                 help="ID of the template")
    # ap.add_argument("--mode=", required=True,
    #                 help="Setup or Extraction")
    # args = vars(ap.parse_args())
    #
    # pdf_name = ast.literal_eval(args['file_path='])
    # claim_list = ast.literal_eval(args['claim_list='])
    # temp_id = ast.literal_eval(args['template_id='])
    # run_mode = args['mode=']

    pdf_name = r'D:\git-workprojects\Work-Projects\CPA_Pipeline_M\New_templates\temp64\43699375_20180514083321335_Capital_claim.pdf'
    claim_list = [1]
    temp_id = 17
    run_mode = "Setup"

    input_excel_M1 = r'D:\git-workprojects\Work-Projects\CPA_Pipeline_M\Non_table_headers.csv'
    input_excel_M2 = r'D:\git-workprojects\Work-Projects\CPA_Pipeline_M\Table_headers.csv'
    m1_path = r'D:\git-workprojects\Work-Projects\CPA_Pipeline_M\M1'
    m2_path = r'D:\git-workprojects\Work-Projects\CPA_Pipeline_M\M2'

    document_object = create_document()
    spacy_object = Spacy_Model.spacy_predictions()
    highlight_object = PDF_Highlight.heuristics()
    saving_headers_object = Saving_In_Excel.save_headers()
    extract_M4_data_object = Model_M4.table_data()
    extract_M3_data_object = Model_M3.extract_data()

    if run_mode == "Setup":

        Document, Merged_Document = document_object.get_Document_Corpus(file_path=pdf_name)

        #if len(Document)>0:
        final_annotated_text = document_object.get_spacy_json_from_Document(Document,claim_list)

        m1_pred = spacy_object.model_m1(final_annotated_text,m1_path)
        m2_pred = spacy_object.model_m2(final_annotated_text,m2_path)

        highlight_object.m1(pdf_name,m1_pred,Document)
        saving_headers_object.save_for_m1(pdf_name,Document,claim_list,temp_id)

        highlight_object.m2(pdf_name,m2_pred,Document,Merged_Document)
        saving_headers_object.save_for_m2(pdf_name,Document,claim_list,temp_id)

    elif run_mode == "Extract":
        Document, Merged_Document = document_object.get_Document_Corpus(file_path=pdf_name)
        extract_M3_data_object.main(pdf_name,input_excel_M1,input_excel_M2,claim_list,Document,temp_id)
        extract_M4_data_object.extract_data(pdf_name, input_excel_M2, Document, Merged_Document, claim_list, temp_id)
    else:
        logger.info("Undefined mode,please give 'Setup' or 'Extract'")