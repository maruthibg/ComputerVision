from configparser import ConfigParser

debug = True
process_videos = True
source_path = 'd:\PROJECTS\maruthi_utils\scanner\videos'
tesseract_command_line = 'C:/Program Files (x86)/Tesseract-OCR/tesseract.exe'


def config(filename='database.ini', section='postgresql'):
    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(filename)

    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception(
            'Section {0} not found in the {1} file'.format(
                section, filename))
    return db
