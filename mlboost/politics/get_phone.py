''' get phone numbers fro 411
cut -d ';' -f 10,11,12,13,14,16,19,23 mtl_steph.csv | awk -F ';'  '{print $2,$1","$4","$3","$5","$6","$7","$8}' |python get_phone.py

https://maps.google.ca/maps?q=Berri+%2F+Ren%C3%A9-L%C3%A9vesque,+Montr%C3%A9al,+QC&hl=fr&ie=UTF8&sll=45.619561,-73.605652&sspn=0.204828,0.506058&oq=berri+rene&hnear=Berri+%2F+Ren%C3%A9-L%C3%A9vesque&t=m&z=17
http://411.ca/search/?q=denis%20lacasse&st=person&point=45.5756,-73.730621&nearme=1&p=1
'''
import re
import sys
import urllib.request, urllib.error, urllib.parse
import time
point = '&point=45.619561,-73.605652&nearme=1&p=1'
url_411="http://411.ca/search/?q={name}&st=person%s" %point
pattern_href = re.compile('.*<a class="styleGU" href="(.*)">... more</a>.*')
pattern_phone = re.compile('.*<h2 class="person_phone_number" itemprop="telephone">(.*)</h2>.*')
pattern_postal = re.compile('.*span itemprop="postalCode">(.*)</span></h2>.*')

def get_phone_number(full_name, postal_code=None):
    try:
        url = url_411.format(name=full_name).replace(" ","%20")
        #print url
        response = urllib.request.urlopen(url)
        content = response.read()
        url = "http://411.ca/%s" %pattern_href.search(content).group(1)
        response = urllib.request.urlopen(url)
        content = response.read()
        postal = pattern_postal.search(content).group(1).replace(' ','')
        if not postal or (postal and postal == postal_code): 
            return pattern_phone.search(content).group(1)
        else:
            print("%s != %s" %(postal, postal_code))
            return "bad"
    except Exception as ex:
        print(ex)
        return None
    
if __name__ == "__main__":
    from optparse import OptionParser
    op = OptionParser(__doc__)
    op.add_option("-n", default=None, dest="name", 
                  help="full name to search")
    op.add_option("-p", default=0, type="int", dest="pos_name", 
                  help="position of full name")
    op.add_option("-P", default=6, type="int", dest="pos_postal", 
                  help="position of postal code")
    op.add_option("-o", default="numbers.csv", dest="output", 
                  help="output fname") 
    op.add_option("-s", default=0, dest="sleep", type=int, 
                  help="output fname")
    opts, args = op.parse_args(sys.argv)
    
    if opts.name:
        print("%s\t%s" %(opts.name, get_phone_number(opts.name)))
    else:
        print("reading from stdin, saving -> %s" %(opts.output))
        out = open(opts.output, 'w')
        for i, line in enumerate(sys.stdin):
            print(i+1)
            line = line.strip()
            els =  line.split(',')
            name = els[opts.pos_name]
            postal = els[opts.pos_postal] 
            number = get_phone_number(name, postal)
            if opts.sleep>0:
                time.sleep(opts.sleep)
            out.write("%s,%s\n" %(line, number))
        out.close()
