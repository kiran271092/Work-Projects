import fitz

pdf = fitz.open(r'C:\Users\kiran.kandula\PycharmProjects\CPA_Pipeline_M\testing_Invoice_20180419_063005487_165480.pdf')

page = pdf[0]

annot = page.firstAnnot

pts = []
comments = {}
while annot is not None:
    if(annot.info['subject'] == "Sticky Note"):
        comments[annot.info['content']] = annot.rect
       # comments.append(annot.rect)
    else:
        pts.append(annot.rect)
    annot = annot.next


print(annot.info['content'])

print(annot.info['content'])

next_annot = annot.next

print(next_annot.info['content'])

pdf.close()

