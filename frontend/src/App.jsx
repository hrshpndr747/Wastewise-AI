import { useEffect, useMemo, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Container,
  LinearProgress,
  Paper,
  Stack,
  Typography,
} from "@mui/material";
import CloudUploadOutlinedIcon from "@mui/icons-material/CloudUploadOutlined";
import RecyclingOutlinedIcon from "@mui/icons-material/RecyclingOutlined";
import RestartAltOutlinedIcon from "@mui/icons-material/RestartAltOutlined";
import { classifyWaste } from "./api";

const acceptedTypes = ["image/jpeg", "image/png", "image/webp"];
const maxSize = 10 * 1024 * 1024;

function titleCase(value) {
  return value
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!file) {
      setPreviewUrl("");
      return undefined;
    }

    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const confidencePercent = useMemo(() => {
    if (!result) return 0;
    return Math.round(result.confidence * 100);
  }, [result]);

  function selectFile(selected) {
    setError("");
    setResult(null);

    if (!selected) return;
    if (!acceptedTypes.includes(selected.type)) {
      setError("Select a JPEG, PNG, or WebP image.");
      return;
    }
    if (selected.size > maxSize) {
      setError("The image must be smaller than 10 MB.");
      return;
    }
    setFile(selected);
  }

  async function submit() {
    if (!file) {
      setError("Choose an image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const payload = await classifyWaste(file);
      setResult(payload);
    } catch (requestError) {
      const detail = requestError.response?.data?.detail;
      setError(
        detail ||
          "Prediction failed. Confirm that the FastAPI server is running.",
      );
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setFile(null);
    setResult(null);
    setError("");
  }

  return (
    <Box className="page-shell">
      <Container maxWidth="md">
        <Stack spacing={3}>
          <Box textAlign="center">
            <RecyclingOutlinedIcon sx={{ fontSize: 58 }} />
            <Typography variant="h3" component="h1" fontWeight={800}>
              EcoVision AI
            </Typography>
            <Typography variant="h6" color="text.secondary">
              Upload one recyclable item and classify its material.
            </Typography>
          </Box>

          <Paper
            className="upload-panel"
            variant="outlined"
            onDragOver={(event) => event.preventDefault()}
            onDrop={(event) => {
              event.preventDefault();
              selectFile(event.dataTransfer.files?.[0]);
            }}
          >
            <Stack spacing={2} alignItems="center">
              <CloudUploadOutlinedIcon sx={{ fontSize: 54 }} />
              <Typography variant="h6">
                Drag and drop an image here
              </Typography>
              <Typography color="text.secondary">or</Typography>
              <Button variant="contained" component="label">
                Choose image
                <input
                  hidden
                  type="file"
                  accept=".jpg,.jpeg,.png,.webp"
                  onChange={(event) => selectFile(event.target.files?.[0])}
                />
              </Button>
              <Typography variant="caption" color="text.secondary">
                JPEG, PNG, or WebP · Maximum 10 MB
              </Typography>
            </Stack>
          </Paper>

          {error && <Alert severity="error">{error}</Alert>}

          {previewUrl && (
            <Card variant="outlined">
              <CardContent>
                <Stack spacing={2} alignItems="center">
                  <Box
                    component="img"
                    src={previewUrl}
                    alt="Selected waste item"
                    className="preview-image"
                  />
                  <Typography variant="body2" color="text.secondary">
                    {file.name}
                  </Typography>
                  <Stack direction={{ xs: "column", sm: "row" }} spacing={2}>
                    <Button
                      variant="contained"
                      onClick={submit}
                      disabled={loading}
                    >
                      {loading ? "Classifying…" : "Classify waste"}
                    </Button>
                    <Button
                      variant="outlined"
                      startIcon={<RestartAltOutlinedIcon />}
                      onClick={reset}
                      disabled={loading}
                    >
                      Reset
                    </Button>
                  </Stack>
                </Stack>
              </CardContent>
              {loading && <LinearProgress />}
            </Card>
          )}

          {loading && (
            <Stack direction="row" spacing={2} justifyContent="center">
              <CircularProgress size={24} />
              <Typography>Running the CNN model…</Typography>
            </Stack>
          )}

          {result && (
            <Card className="result-card" variant="outlined">
              <CardContent>
                <Stack spacing={2}>
                  <Alert
                    severity={result.status === "success" ? "success" : "warning"}
                  >
                    {result.status === "success"
                      ? "Classification completed."
                      : "The model is not confident enough to classify this image."}
                  </Alert>

                  <Box>
                    <Typography variant="overline" color="text.secondary">
                      Prediction
                    </Typography>
                    <Typography variant="h4" fontWeight={800}>
                      {result.predicted_class
                        ? titleCase(result.predicted_class)
                        : "Uncertain"}
                    </Typography>
                  </Box>

                  <Box>
                    <Stack
                      direction="row"
                      justifyContent="space-between"
                      alignItems="center"
                    >
                      <Typography>Confidence</Typography>
                      <Chip label={`${confidencePercent}%`} />
                    </Stack>
                    <LinearProgress
                      variant="determinate"
                      value={confidencePercent}
                      sx={{ mt: 1, height: 10, borderRadius: 5 }}
                    />
                  </Box>

                  <Typography>{result.recommendation}</Typography>

                  <Box>
                    <Typography variant="subtitle2" gutterBottom>
                      Top model scores
                    </Typography>
                    <Stack direction="row" flexWrap="wrap" gap={1}>
                      {result.top_predictions.map((prediction) => (
                        <Chip
                          key={prediction.class_name}
                          variant="outlined"
                          label={`${titleCase(prediction.class_name)} ${Math.round(
                            prediction.confidence * 100,
                          )}%`}
                        />
                      ))}
                    </Stack>
                  </Box>
                </Stack>
              </CardContent>
            </Card>
          )}

          <Typography
            variant="caption"
            textAlign="center"
            color="text.secondary"
          >
            Educational prototype. Always follow local recycling rules.
          </Typography>
        </Stack>
      </Container>
    </Box>
  );
}
