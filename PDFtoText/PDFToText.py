from PyPDF4 import PdfFileReader
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
import io


def pdfparser(data):
    fp = open(data, 'rb')
    rsrcmgr = PDFResourceManager()
    retstr = io.StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    # Create a PDF interpreter object.
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    # Process each page contained in the document.
    pages = PdfFileReader(open(data, "rb"), strict=False).getNumPages()
    i = 0
    for page in PDFPage.get_pages(fp):
        i += 1
        if i > pages - 2:
            break
        interpreter.process_page(page)
        data = retstr.getvalue()
    file = open("textTA.txt", "wb")
    file.write(data.encode())
    file.close()


def getquestion():
    file = open("textTA.txt", 'r')
    result = []
    ques = []
    rs = []
    for line in file:
        line = line.strip()
        if len(line) == 0 :
            continue
        if line.find('次の') >= 0:
            if len(ques) == 0:
                ques.append(line.replace('\u3000', ' '))
            else:
                result.append([ques, rs])
                ques = [line.replace('\u3000', ' ')]
                rs = []
        else:
            if len(ques) > 0:
                rs.append(line.replace('\u3000', ' '))
    result.append([ques, rs])
    return result

def getquestiontxt():
    file = open("textTA.txt", 'r')
    result = ''
    ques = ''
    rs = ''
    for line in file:
        line = line.strip()
        if len(line) == 0 :
            continue
        if line.find('次の') >= 0:
            if len(ques) == 0:
                ques = line.replace('\u3000', ' ') + '\n'
            else:
                result += ques + rs + '\n'
                ques = line.replace('\u3000', ' ') + '\n'
                rs = ''
        else:
            if len(ques) > 0:
                rs += line.replace('\u3000', ' ') + '\n'
    result += ques + rs + '\n'
    return result + '\n'


if __name__ == '__main__':
    name = ['TA1', 'TA2', 'TA3', 'TA4', 'TA5', 'TA6', 'TA7', 'TA8']
    file = open("result.txt", "wb")
    for fname in name:
        pdfparser(fname + '.pdf')
        data = getquestiontxt()
        file.write(data.encode())



