import { useParams } from "react-router-dom";

export default function RunDetailPage() {
  const { summaryPath } = useParams<{ summaryPath: string }>();
  return <div>Run Detail: {summaryPath ? decodeURIComponent(summaryPath) : "Loading..."}</div>;
}
