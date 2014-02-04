import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;
 
import java.io.File;
import java.io.IOException;
/**
 * Java file for doing simple preprocessing stuff...
 *
 */
public class Preprocessor {
  /**
   * takes 2 arguments:
   * - CSV input file
   * - ARFF output file
   */
  public static void main(String[] args) throws Exception {
    if (args.length != 2) {
      System.out.println("\nUsage: Preprocess <input.csv> <output>\n");
      System.exit(1);
    }
 
    // load CSV
    CSVLoader loader = new CSVLoader();
    loader.setSource(new File(args[0]));
    Instances data = loader.getDataSet();
 
    System.out.println("Read original Dataset. Size: " + data.numInstances());
    
    //first randomize all
    Randomize r = new Randomize();
    r.setInputFormat(data);
    r.setRandomSeed(42);
    Instances randomized = Filter.useFilter(data, r);
    writeArffToFile(args[1], "_randomized_all", randomized);
    
    System.out.println("Randomized Size: " + randomized.numInstances());
    
    //then use mu sigma standardization
    Standardize s = new Standardize();
    s.setInputFormat(randomized);
    Instances standardized = Filter.useFilter(randomized, s);
    writeArffToFile(args[1], "_standardized_all", standardized);
    
    System.out.println("Standardized Size: " + standardized.numInstances());
    
    //afterwards do the training test split
    RemovePercentage rp = new RemovePercentage();
    rp.setInvertSelection(false);
    rp.setPercentage(10); //set 90:10 split
    rp.setInputFormat(standardized);
    
    Instances training = Filter.useFilter(standardized, rp);
    System.out.println("Training Size: " + training.numInstances());
    
    
    rp = new RemovePercentage();
    rp.setInputFormat(standardized);
    rp.setInvertSelection(true);
    rp.setPercentage(10);
    
    Instances test = Filter.useFilter(standardized, rp);
    System.out.println("Test Size: " + test.numInstances());
   
    writeArffToFile("", "TRAINING", training);
    writeArffToFile("", "TEST",test);
    
    System.out.println("Have fun with the project!!");
  }
  
  public static void writeArffToFile(String nameTemplate, String nameAdd, Instances data) throws IOException{
	// save ARFF
	    ArffSaver saver = new ArffSaver();
	    saver.setInstances(data);
	    saver.setFile(new File(nameTemplate + nameAdd + ".arff"));
	    saver.setDestination(new File(nameTemplate + nameAdd + ".arff"));
	    saver.writeBatch();
  }
}
