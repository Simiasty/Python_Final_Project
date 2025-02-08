import configparser

def create_config():
    config = configparser.ConfigParser()
    config['General'] = {'fisher': False, 'paradigm': 'story', 'output_folder': "Figures"}
    config['Languages'] = {
        'language_list': ", ".join([
            "Armenian", "Irish", "Greek", "Catalan", "French", "Italian", "Portuguese", "Romanian", "Spanish", 
            "Afrikaans", "Danish", "Dutch", "English", "German", "Norwegian", "Swedish", "Belarusian", "Bulgarian",
            "Czech", "Latvian", "Lithuanian", "Polish", "Russian", "Serbocroatian", "Slovene", "Ukrainian", "Farsi",
            "Gujarati", "Hindi", "Marathi", "Nepali", "Arabic", "Hebrew", "Vietnamese", "Tagalog", "Tamil", "Telugu",
            "Japanese", "Korean", "Swahili", "Mandarin", "Finnish", "Hungarian", "Turkish", "Basque"
        ])
    }
    with open('config.ini', 'w') as configfile:
        config.write(configfile)

def read_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return {
        'fisher': config.getboolean('General', 'fisher'),
        'paradigm': config.get('General', 'paradigm'),
        'output_folder': config.get('General', 'output_folder'),
        'language_list': config.get('Languages', 'language_list').split(', ')
    }