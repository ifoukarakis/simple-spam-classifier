package gr.ifoukarakis.spam;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.sql.*;

import java.io.IOException;
import java.time.Instant;

public class LogisticRegressionModel {
    private SparkSession spark;

    public LogisticRegressionModel(JavaSparkContext sc) {
        SQLContext context = new SQLContext(sc);
        this.spark = context.sparkSession();
    }

    public void run(JavaSparkContext sc, JavaRDD<Email> emails) throws IOException {
        Dataset<Email> dataset = this.spark.createDataset(emails.rdd(), Encoders.bean(Email.class));
        // Split dataset to train-test
        Dataset<Email>[] datasets = dataset.randomSplit(new double[]{0.8, 0.2});

        // Create the training pipeline
        Tokenizer tokenizer = new Tokenizer().setInputCol("body").setOutputCol("words");
        HashingTF tf = new HashingTF().setInputCol(tokenizer.getOutputCol()).setOutputCol("features");
        LogisticRegression lr = new LogisticRegression().setMaxIter(10).setRegParam(0.01);
        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] {tokenizer, tf, lr});

        // Train the model
        PipelineModel model = pipeline.fit(datasets[0]);

        // Predict on test dataset
        Dataset<Row> prediction = model.transform(datasets[1]);

        // Evaluate using a set of metrics
        MulticlassClassificationEvaluator eval1 = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        MulticlassClassificationEvaluator eval2 = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("weightedPrecision");

        MulticlassClassificationEvaluator eval3 = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("weightedRecall");

        System.out.println(String.format("Accuracy = %s", eval1.evaluate(prediction)));
        System.out.println(String.format("Precision = %s", eval2.evaluate(prediction)));
        System.out.println(String.format("Recall = %s", eval3.evaluate(prediction)));
    }

    public static void main(String[] args) throws IOException {
        SparkConf conf = new SparkConf()
                .setAppName("Spam or ham Detection")
                .setMaster("local[*]")
                .set("spark.driver.maxResultSize", "4g");
        JavaSparkContext sc = new JavaSparkContext(conf);

        JavaRDD<Email> spam = sc.textFile("enron1/spam/*.txt").map(x -> new Email(x, 1));
        JavaRDD<Email> ham = sc.textFile("enron1/ham/*.txt").map(x -> new Email(x, 0));
        JavaRDD<Email> emails = spam.union(ham);

        LogisticRegressionModel model = new LogisticRegressionModel(sc);
        model.run(sc, emails);
    }
}
