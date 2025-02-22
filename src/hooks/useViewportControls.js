import { useEffect } from "react";

const useViewportControls = () => {
  useEffect(() => {
    console.log("Viewport controls activated.");
  }, []);
};

export default useViewportControls;
