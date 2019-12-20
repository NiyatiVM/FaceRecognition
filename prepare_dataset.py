#all the modules to be imported
from requests import exceptions
import argparse
import requests
import cv2
import os

mainpath=r'C:\Users\Niyati\Desktop\FaceRecog\Phase-01'
directory='train_dir'
dir_path = os.path.join(mainpath,directory)
if(os.path.exists(dir_path)):
	pass
else:
	os.mkdir(dir_path)
#since the searchitem and path to store results will be specified at runtime ,we use argsparse for cmd arguments
ap=argparse.ArgumentParser()
ap.add_argument("-q","--query",required=True,help="search query to search Bing Image API for")
args=vars(ap.parse_args())
output=os.path.join(dir_path,args["query"])
os.mkdir(output)

#connection to microsoft bing search API 
#API can be obtained from 
API_KEY ="YOUR API KEY"
MAX_RESULTS = 300
GROUP_SIZE = 60

URL="https://api.cognitive.microsoft.com/bing/v7.0/images/search"

#Handling the exceptions
EXCEPTIONS = set([IOError, FileNotFoundError,exceptions.RequestException,exceptions.HTTPError,exceptions.ConnectionError,exceptions.Timeout])
 
#Search begins :
#Refer to https://docs.microsoft.com/en-us/rest/api/cognitiveservices-bingsearch/bing-images-api-v7-reference
#Initialize parameters
term=args["query"]
headers ={"Ocp-Apim-Subscription-Key":API_KEY}
params = {"q" :term,"offset":0,"count":GROUP_SIZE}

#Search
print("[INFO] searching Bing API for '{}'".format(term))
search = requests.get(URL, headers=headers, params=params)
search.raise_for_status()

#store results
results = search.json()
estNumResults =min(results["totalEstimatedMatches"],MAX_RESULTS)
print("[INFO] {} total reslts for '{}'".format(estNumResults,term))

total = 0
for offset in range(0, estNumResults, GROUP_SIZE):
	# update the search parameters using the current offset, then
	# make the request to fetch the results
	print("[INFO] making request for group {}-{} of {}...".format(
		offset, offset + GROUP_SIZE, estNumResults))
	params["offset"] = offset
	search = requests.get(URL, headers=headers, params=params)
	search.raise_for_status()
	results = search.json()
	print("[INFO] saving images for group {}-{} of {}...".format(
		offset, offset + GROUP_SIZE, estNumResults))
	for v in results["value"]:
		try:
			print("[INFO] fetching : {}".format(v["contentUrl"]))
			r = requests.get(v["contentUrl"],timeout=30)
			
			ext = v["contentUrl"][v["contentUrl"].rfind("."):]
			p = os.path.sep.join([output, "{}{}".format(
					str(total).zfill(8), ext)])
			
			# write the image to disk
			f = open(p, "wb")
			f.write(r.content)
			f.close
		
		#exceptions
		except Exception as e:
			if type(e) in EXCEPTIONS:
					print("[INFO] skipping: {}".format(v["contentUrl"]))
					continue
			
		image = cv2.imread(p)
	 
		"""if image is None:
			print("[INFO] deleting: {}".format(p))
			os.remove(p)
			continue"""
	 
		total += 1