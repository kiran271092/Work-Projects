
import warnings
import spacy
import pandas as pd
warnings.filterwarnings('ignore')

class spacy_predictions:

    ##Newly added function to get the unique headers for multipage claim
    def get_unique(self,output):#rename the output variable to m1_output
        cols = ['Page Key', 'Labels', 'Start', 'End']
        df = pd.DataFrame(columns=cols)
        key_list = list(output.keys())

        for key in key_list:
            content_keys = output[key]['Header'].keys()
            for content in content_keys:
                if len(df[df['Labels'].str.contains(content)]) == 0:
                    temp_df = pd.DataFrame(columns=cols)
                    temp_df.loc[0, 'Page Key'] = key
                    temp_df.loc[0, 'Labels'] = content
                    temp_df.loc[0, 'Start'] = output[key]['Header'][content][0]
                    temp_df.loc[0, 'End'] = output[key]['Header'][content][1]
                    df = df.append(temp_df)

        key_list = list(df['Page Key'].unique())
        page_output = {}
        for key in key_list:
            head = {}
            temp_df = df[df['Page Key'] == key]
            content = {}
            for index, cell in temp_df.iterrows():
                pos = []
                pos.append(cell['Start'])
                pos.append(cell['End'])
                content[cell['Labels']] = pos
            head['Header'] = content
            page_output[key] = head
        return page_output

    def model_m1(self,json_input,model_path):
        
        nlp2 = spacy.load(model_path)
        keys = list(json_input.keys())
        page = {}
        for key in keys:
            d={}
            json_data = json_input[key]
            doc_to_text = nlp2(json_data['content'])

            #exception needs to be handled when ever doc_to_text is null;

            if len(doc_to_text.ents)>0:
                d[doc_to_text.ents[0].label_] = {}
                for ent in doc_to_text.ents:
                    d[ent.label_][ent.text] = [ent.start_char,ent.end_char]

            else:
                d = {}
            page[key] = d
        try:

            page = self.get_unique(page)

        except:

            page


        return page  
    
    
    def model_m2(self,json_input,model_path):
        
        nlp2 = spacy.load(model_path)
        keys = list(json_input.keys())
        # if 'ner' not in nlp2.pipe_names:
        #     ner = nlp2.create_pipe('ner')
        #     nlp2.add_pipe(ner, last=True)
        page_data = {}
        for key in keys:
            json_data = json_input[key]
            doc_to_text = nlp2(json_data['content'])
            header_path = []
            for ent in doc_to_text.ents:
                header_path.append(ent.text)
            page_data[key] = header_path
        return page_data

