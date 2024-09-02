import React, { useState } from "react";

export default function Canvas() {
  const [mouseDown, setMouseDown] = useState(false);
  return (
    <>
      <canvas
        id="myCanvas"
        width="800"
        height="800"
        style={{ border: "solid", solid: "#000000" }}
        onMouseDown={() => setMouseDown(true)}
        onMouseUp={() => setMouseDown(false)}
        onMouseMove={(e) => {
          if (mouseDown) {
            const canvas = document.getElementById("myCanvas");
            const ctx = canvas.getContext("2d");
            ctx.fillStyle = "#FF0000";
            ctx.fillRect(e.nativeEvent.offsetX, e.nativeEvent.offsetY, 5, 5);
          }
        }}
      ></canvas>
    </>
  );
}
