using Microsoft.ML.Runtime.Api;

namespace serverlessfunctionapp.Predict
{
    public class IrisPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabels;
    }
}