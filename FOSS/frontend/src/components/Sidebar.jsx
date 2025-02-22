import React from "react";
import { Button, Nav } from "react-bootstrap";

const Sidebar = ({ onMove }) => {
  return (
    <Nav className="d-flex flex-column p-3 bg-light" style={{ height: "100vh" }}>
      <h4 className="text-center mb-3">Tools</h4>

      <h5 className="text-center">Move</h5>
      <Button variant="info" className="mb-2" onClick={() => onMove("left")}>
        Left
      </Button>
      <Button variant="info" className="mb-2" onClick={() => onMove("right")}>
        Right
      </Button>
      <Button variant="info" className="mb-2" onClick={() => onMove("up")}>
        Up
      </Button>
      <Button variant="info" className="mb-2" onClick={() => onMove("down")}>
        Down
      </Button>
    </Nav>
  );
};

export default Sidebar;
