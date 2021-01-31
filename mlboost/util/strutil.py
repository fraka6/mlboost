
def str2perc(s):
    '''optimal string formating to percentage'''
    if s=="" or s.isalnum() or not s[1:].replace('.','').isdigit():
        return s
    else:
        s = "%2.2f" %float(s)
        if len(s)<6:
            return " "*(6-len(s))+s
        else:
            return s
