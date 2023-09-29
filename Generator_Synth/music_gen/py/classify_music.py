from util.classifier import Classifier
import argparse as arg

parser = arg.ArgumentParser()

parser.add_argument("-i", "--input", help="input path to classify music.", name_or_flags="input")
parser.add_argument("-o", "--output", help="output directory to classify music.", name_or_flags="output")
parser.add_argument("-n", "--name", help="name of the music to classify.", default="music", name_or_flags="name")

def classify_music(path: str) -> Classifier:
    # Classify music using YAMNet
    classifier = Classifier()
    classifier.classify_single(path)
    return classifier

if __name__ == "__main__":
    args = parser.parse_args()
    classifier = classify_music(args.input)
    classifier.to_csv(args.output + "\\" + args.name + ".csv")
