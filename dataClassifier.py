# dataClassifier.py
# -----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# This file contains feature extraction methods and harness
# code for data classification

import perceptron
import samples
import sys
import util


TEST_SET_SIZE = 100
DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
FACE_DATUM_WIDTH=60
FACE_DATUM_HEIGHT=70


def basicFeatureExtractorDigit(datum):
    """
    Returns a set of pixel features indicating whether
    each pixel in the provided datum is white (0) or gray/black (1)
    """
    a = datum.getPixels()

    features = util.Counter()
    for x in range(DIGIT_DATUM_WIDTH):
        for y in range(DIGIT_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x,y)] = 1
            else:
                features[(x,y)] = 0
    return features



def enhancedFeatureExtractorDigit(datum):
    """
    Your feature extraction playground.

    You should return a util.Counter() of features
    for this datum (datum is of type samples.Datum).

    ## DESCRIBE YOUR ENHANCED FEATURES HERE...
    #some enhanced features I note are:
        -Continuous Region. (1,2,3,5,7) Have one region of whitespace 
         (6,8,9,0) have more regions, (4) might have 1 or two.
        - Long Straight Edges. (1,4,5,7,9) will likely have one long straight edge 
          (0,2,3,6,8) will likely not. Determined by having n in a row or diagonal plus signs
    ##
    """


    features =  basicFeatureExtractorDigit(datum)
    #features = util.Counter()

    "*** YOUR CODE HERE ***"
    a = datum.getPixels() 


    #determine whether has vertical straight edge
    horizontal_counts= []
    for y in range(DIGIT_DATUM_HEIGHT):
        s= 0
        t= 0
        while s < DIGIT_DATUM_WIDTH:
            count = 0
            if datum.getPixel(s, y) == 1:
                count +=1
                while t+1< DIGIT_DATUM_WIDTH:
                    t+=1
                    if datum.getPixel(t, y) == 1:
                        count +=1
                    else:
                        horizontal_counts.append(count)
                        s=t
                        break
        
            s+=1
            t+=1
    flats= filter(lambda x: x> 5, horizontal_counts)
    #print(flats)
    #print(datum)

    vertical_counts= []
    for x in range(DIGIT_DATUM_WIDTH):
        s= 0
        t= 0
        while s < DIGIT_DATUM_HEIGHT:
            count = 0
            if datum.getPixel(x, s) == 1:
                count +=1
                while t+1< DIGIT_DATUM_HEIGHT:
                    t+=1
                    if datum.getPixel(x, t) == 1:
                        count +=1
                    else:
                        vertical_counts.append(count)
                        s=t
                        break
        
            s+=1
            t+=1
    straights= filter(lambda x: x> 4, vertical_counts)
    #print(straights)
    #print(datum)


    #determine percentage of nonzeros
    nonzeros = 0
    #determine top-heavy or bottom-heavy
    top = 0
    bottom= 0
    for x in range(DIGIT_DATUM_WIDTH):
        for y in range(DIGIT_DATUM_HEIGHT):
            if y <= DIGIT_DATUM_HEIGHT/2:
                if datum.getPixel(x, y) > 0:
                    bottom +=1
            else:
                if datum.getPixel(x, y) > 0:
                    top +=1
    nonzeros = top + bottom

    for i in range(5):
        features["straight"+ str(i)] = len(straights) > i #there is 1,2,3... straights > len(5)
    
    for i in range(5):
        features["flats"+ str(i)] = len(flats) > i #there is 1,2,3... straights > len(5)

    ###____arbitrary feature___###
    """num_pixels= DIGIT_DATUM_WIDTH * DIGIT_DATUM_HEIGHT
                for i in range(5):
                    features["fill" + str(i)] = nonzeros > 100 #fill 10, 15, 20,.. percent
                """

    l= [top,bottom]
    #print(l)
    #for p in range(5):
    for i in range(len(l)):
        features["heaviness" + str(i)] = l[i%2] > l[(i+1)%2]
        #print(i, ":", features["heaviness" + str(i)])
    return features




## =====================
## You don't have to modify any code below.
## =====================


class ImagePrinter:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def printImage(self, pixels):
        """
        Prints a Datum object that contains all pixels in the
        provided list of pixels.  This will serve as a helper function
        to the analysis function you write.

        Pixels should take the form
        [(2,2), (2, 3), ...]
        where each tuple represents a pixel.
        """
        image = samples.Datum(None,self.width,self.height)
        for pix in pixels:
            try:
            # This is so that new features that you could define which
            # which are not of the form of (x,y) will not break
            # this image printer...
                x,y = pix
                image.pixels[x][y] = 2
            except:
                print "new features:", pix
                continue
        print image

def default(str):
    return str + ' [Default: %default]'

USAGE_STRING = """
  USAGE:      python dataClassifier.py <options>
  EXAMPLES:   (1) python dataClassifier.py
                  - trains the default mostFrequent classifier on the digit dataset
                  using the default 100 training examples and
                  then test the classifier on test data
              (2) python dataClassifier.py -c naiveBayes -d digits -t 1000 -f -o -1 3 -2 6 -k 2.5
                  - would run the naive Bayes classifier on 1000 training examples
                  using the enhancedFeatureExtractorDigits function to get the features
                  on the faces dataset, would use the smoothing parameter equals to 2.5, would
                  test the classifier on the test data and performs an odd ratio analysis
                  with label1=3 vs. label2=6
                 """


def readCommand( argv ):
    "Processes the command used to run from the command line."
    from optparse import OptionParser
    parser = OptionParser(USAGE_STRING)

    parser.add_option('-c', '--classifier', help=default('The type of classifier'), choices=['mostFrequent', 'nb', 'naiveBayes', 'perceptron', 'mira', 'minicontest'], default='mostFrequent')
    parser.add_option('-d', '--data', help=default('Dataset to use'), choices=['digits', 'faces', 'pacman'], default='digits')
    parser.add_option('-t', '--training', help=default('The size of the training set'), default=100, type="int")
    parser.add_option('-f', '--features', help=default('Whether to use enhanced features'), default=False, action="store_true")
    parser.add_option('-o', '--odds', help=default('Whether to compute odds ratios'), default=False, action="store_true")
    parser.add_option('-1', '--label1', help=default("First label in an odds ratio comparison"), default=0, type="int")
    parser.add_option('-2', '--label2', help=default("Second label in an odds ratio comparison"), default=1, type="int")
    parser.add_option('-w', '--weights', help=default('Whether to print weights'), default=False, action="store_true")
    parser.add_option('-k', '--smoothing', help=default("Smoothing parameter (ignored when using --autotune)"), type="float", default=2.0)
    parser.add_option('-a', '--autotune', help=default("Whether to automatically tune hyperparameters"), default=False, action="store_true")
    parser.add_option('-i', '--iterations', help=default("Maximum iterations to run training"), default=3, type="int")
    parser.add_option('-s', '--test', help=default("Amount of test data to use"), default=TEST_SET_SIZE, type="int")
    parser.add_option('-g', '--agentToClone', help=default("Pacman agent to copy"), default=None, type="str")

    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))
    args = {}

    # Set up variables according to the command line input.
    print "Doing classification"
    print "--------------------"
    print "data:\t\t" + options.data
    print "classifier:\t\t" + options.classifier
    if not options.classifier == 'minicontest':
        print "using enhanced features?:\t" + str(options.features)
    else:
        print "using minicontest feature extractor"
    print "training set size:\t" + str(options.training)
    if(options.data=="digits"):
        printImage = ImagePrinter(DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT).printImage
        if (options.features):
            featureFunction = enhancedFeatureExtractorDigit
        else:
            featureFunction = basicFeatureExtractorDigit
        if (options.classifier == 'minicontest'):
            featureFunction = contestFeatureExtractorDigit
    else:
        print "Unknown dataset", options.data
        print USAGE_STRING
        sys.exit(2)

    if(options.data=="digits"):
        legalLabels = range(10)

    if options.training <= 0:
        print "Training set size should be a positive integer (you provided: %d)" % options.training
        print USAGE_STRING
        sys.exit(2)

    if options.smoothing <= 0:
        print "Please provide a positive number for smoothing (you provided: %f)" % options.smoothing
        print USAGE_STRING
        sys.exit(2)

    if options.odds:
        if options.label1 not in legalLabels or options.label2 not in legalLabels:
            print "Didn't provide a legal labels for the odds ratio: (%d,%d)" % (options.label1, options.label2)
            print USAGE_STRING
            sys.exit(2)

    if(options.classifier == "mostFrequent"):
        classifier = mostFrequent.MostFrequentClassifier(legalLabels)
    elif(options.classifier == "perceptron"):
        if options.data != 'pacman':
            classifier = perceptron.PerceptronClassifier(legalLabels,options.iterations)
    else:
        print "Unknown classifier:", options.classifier
        print USAGE_STRING

        sys.exit(2)

    args['agentToClone'] = options.agentToClone

    args['classifier'] = classifier
    args['featureFunction'] = featureFunction
    args['printImage'] = printImage

    return args, options

# Main harness code

def runClassifier(args, options):
    featureFunction = args['featureFunction']
    classifier = args['classifier']
    printImage = args['printImage']
    
    # Load data
    numTraining = options.training
    numTest = options.test


    rawTrainingData = samples.loadDataFile("digitdata/trainingimages", numTraining,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", numTraining)
    rawValidationData = samples.loadDataFile("digitdata/validationimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    validationLabels = samples.loadLabelsFile("digitdata/validationlabels", numTest)
    rawTestData = samples.loadDataFile("digitdata/testimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("digitdata/testlabels", numTest)


    # Extract features
    print "Extracting features..."
    trainingData = map(featureFunction, rawTrainingData)
    validationData = map(featureFunction, rawValidationData)
    testData = map(featureFunction, rawTestData)

    # Conduct training and testing
    print "Training..."
    classifier.train(trainingData, trainingLabels, validationData, validationLabels)
    print "Validating..."
    guesses = classifier.classify(validationData)
    correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
    print str(correct), ("correct out of " + str(len(validationLabels)) + " (%.1f%%).") % (100.0 * correct / len(validationLabels))
    print "Testing..."
    guesses = classifier.classify(testData)
    correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
    print str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels))
    


    if((options.weights) & (options.classifier == "perceptron")):
        for l in classifier.legalLabels:
            features_weights = classifier.findHighWeightFeatures(l)
            print ("=== Features with high weight for label %d ==="%l)
            printImage(features_weights)

if __name__ == '__main__':
    # Read input
    args, options = readCommand( sys.argv[1:] )
    # Run classifier
    runClassifier(args, options)
