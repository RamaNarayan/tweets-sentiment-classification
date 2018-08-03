import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.feature.{Tokenizer}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

object SentimentAnalysis {
  def writeMetricsToFile(predictionAndLabels: RDD[(Double, Double)],printContent: StringBuilder,modelName: String ) : StringBuilder = {

    // Instantiate metrics object
    val metrics = new MulticlassMetrics(predictionAndLabels)
    printContent.append("\n")
    // Confusion matrix
    printContent.append("Classification Model: "+modelName+"\n")
    printContent.append("Confusion matrix:")
    printContent.append("\n")
    printContent.append(metrics.confusionMatrix)
    printContent.append("\n")
    // Overall Statistics
    val accuracy = metrics.accuracy
    printContent.append("Summary Statistics")
    printContent.append("\n")
    printContent.append(s"Accuracy = $accuracy")
    printContent.append("\n")
    // Precision by label
    val labels = metrics.labels
    labels.foreach { l =>
      printContent.append(s"Precision($l) = " + metrics.precision(l))
      printContent.append("\n")
    }

    // Recall by label
    labels.foreach { l =>
      printContent.append(s"Recall($l) = " + metrics.recall(l))
      printContent.append("\n")
    }

    // False positive rate by label
    labels.foreach { l =>
      printContent.append(s"FPR($l) = " + metrics.falsePositiveRate(l))
      printContent.append("\n")
    }

    // F-measure by label
    labels.foreach { l =>
      printContent.append(s"F1-Score($l) = " + metrics.fMeasure(l))
      printContent.append("\n")
    }

    // Weighted stats
    printContent.append(s"Weighted precision: ${metrics.weightedPrecision}")
    printContent.append("\n")
    printContent.append(s"Weighted recall: ${metrics.weightedRecall}")
    printContent.append("\n")
    printContent.append(s"Weighted F1 score: ${metrics.weightedFMeasure}")
    printContent.append("\n")
    printContent.append(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}")
    printContent.append("\n")
    return printContent
  }
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder.appName("Simple Application").getOrCreate()
    if (args.length != 2) {
      println("I/p and O/p filepath needed")
    }
    Logger.getLogger("labAssignment").setLevel(Level.OFF)
    spark.sparkContext.setLogLevel("ERROR")
    import spark.implicits._
    val tweetData = spark.read.option("header","true")
      .csv(args(0))
    val cols = Array("text")
    val filteredTweetData = tweetData.na.drop(cols)
    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")

    val remover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("filtered")

    val hashingTF = new HashingTF()
      .setInputCol(remover.getOutputCol)
      .setOutputCol("features")

    val indexer = new StringIndexer()
      .setInputCol("airline_sentiment")
      .setOutputCol("categoryIndex")

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover,hashingTF, indexer))

    val preProcessedData = pipeline.fit(filteredTweetData)
    val tweetPreProcessedData = preProcessedData.transform(filteredTweetData)
    val Array(training, test) = tweetPreProcessedData.randomSplit(Array(0.8, 0.2))
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setLabelCol("categoryIndex")
      .setFeaturesCol("features")

    val lr_paramGrid = new ParamGridBuilder()
      .addGrid(hashingTF.numFeatures, Array(10, 50, 100))
      .addGrid(lr.regParam, Array(0.1,0.3))
      .build()

    val lr_evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("categoryIndex")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val lr_cv = new CrossValidator()
      .setEstimator(lr)
      .setEvaluator(lr_evaluator)
      .setEstimatorParamMaps(lr_paramGrid)
      .setNumFolds(3)

    val lr_model = lr_cv.fit(training)

    val lr_prediction = lr_model.transform(test)
    //convert dataset to RDD[(Double,Double)]
    val lr_result = lr_prediction.select("categoryIndex","prediction").map{ case Row(l:Double,p:Double) => (l,p) }

    val lr_predictionAndLabels = lr_result.rdd
    var printContent = new StringBuilder()
    printContent = writeMetricsToFile(lr_predictionAndLabels,printContent,"Logistic Regression")

    val bayes_model = new NaiveBayes()
      .setLabelCol("categoryIndex")
      .setFeaturesCol("features")

    val bm_paramGrid = new ParamGridBuilder()
      .addGrid(hashingTF.numFeatures, Array(10,50,100))
      .addGrid(bayes_model.smoothing,Array(0.1,0.2,0.3))
      .build()

    val bm_evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("categoryIndex")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val bm_cv = new CrossValidator()
      .setEstimator(bayes_model)
      .setEvaluator(bm_evaluator)
      .setEstimatorParamMaps(bm_paramGrid)
      .setNumFolds(3)

    val bmModel = bm_cv.fit(training)

    val bm_prediction = bmModel.transform(test)
    //convert dataset to RDD[(Double,Double)]
    val bm_result = bm_prediction.select("categoryIndex","prediction").map{ case Row(l:Double,p:Double) => (l,p) }

    val bm_predictionAndLabels = bm_result.rdd

    printContent = writeMetricsToFile(bm_predictionAndLabels,printContent,"Naive Bayes")
    val printRdd = spark.sparkContext.parallelize(Seq(printContent))
    printRdd.saveAsTextFile(args(1))
  }
}
