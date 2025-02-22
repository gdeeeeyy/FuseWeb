import React, { useRef } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";

const Cube = ({ position, onMove }) => {
  const meshRef = useRef();

  return (
    <mesh ref={meshRef} position={position}>
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial color="royalblue" />
    </mesh>
  );
};

const Viewport = ({ position }) => {
  return (
    <Canvas camera={{ position: [5, 5, 5] }} style={{ width: "100%", height: "100vh", background: "#ddd" }}>
      <ambientLight intensity={0.5} />
      <directionalLight position={[5, 5, 5]} intensity={1} />
      <Cube position={position} />
      <OrbitControls />
    </Canvas>
  );
};

export default Viewport;
