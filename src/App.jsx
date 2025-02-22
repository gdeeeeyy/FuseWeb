import React, { useState } from "react";
import { Container, Row, Col } from "react-bootstrap";
import Sidebar from "./components/Sidebar";
import Toolbar from "./components/Toolbar";
import Viewport from "./components/Viewport";

const App = () => {
  const [position, setPosition] = useState([0, 0, 0]);

  const handleMove = (direction) => {
    setPosition((prev) => {
      const [x, y, z] = prev;
      const moveStep = 0.5;
      const maxLimit = 5;
      const minLimit = -5;

      switch (direction) {
        case "left":
          return [Math.max(x - moveStep, minLimit), y, z];
        case "right":
          return [Math.min(x + moveStep, maxLimit), y, z];
        case "up":
          return [x, Math.min(y + moveStep, maxLimit), z];
        case "down":
          return [x, Math.max(y - moveStep, minLimit), z];
        default:
          return prev;
      }
    });
  };

  return (
    <Container fluid>
      <Row>
        <Col md={2} className="bg-light">
          <Sidebar onMove={handleMove} />
        </Col>
        <Col md={10}>
          <Toolbar />
          <Viewport position={position} />
        </Col>
      </Row>
    </Container>
  );
};

export default App;
