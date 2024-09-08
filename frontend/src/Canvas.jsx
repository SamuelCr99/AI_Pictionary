import React, { useState, useEffect, useRef } from "react";

export default function Canvas() {
  const [mouseDown, setMouseDown] = useState(false);
  const [prevPos, setPrevPos] = useState({ x: null, y: null });

  const handleMouseMove = (e) => {
    if (mouseDown) {
      const currentX = e.nativeEvent.offsetX;
      const currentY = e.nativeEvent.offsetY;
      const canvas = document.getElementById("myCanvas");
      const ctx = canvas.getContext("2d");

      if (prevPos.x !== null && prevPos.y !== null) {
        ctx.lineWidth = 20;
        ctx.beginPath();
        ctx.moveTo(prevPos.x, prevPos.y);
        ctx.lineTo(currentX, currentY);
        ctx.stroke();
        ctx.closePath();
        // console.log(ctx.getImageData(0, 0, 800, 800));
      }

      setPrevPos({ x: currentX, y: currentY });
    }
  };

  const handleMouseUp = () => {
    setMouseDown(false);
    setPrevPos({ x: null, y: null }); // Reset previous position on mouse up
  };

  return (
    <>
      <canvas
        id="myCanvas"
        width="800"
        height="800"
        style={{ border: "solid", borderColor: "#FF0000" }}
        onMouseDown={() => setMouseDown(true)}
        onMouseUp={handleMouseUp}
        onMouseMove={handleMouseMove}
      ></canvas>
      <canvas
        id="smallCanvas"
        width="28"
        height="28"
        style={{ border: "solid", borderColor: "#FF0000" }}
      ></canvas>
      <button
        onClick={() => {
          const canvas = document.getElementById("myCanvas");
          const ctx = canvas.getContext("2d");

          // // Create a new 28x28 canvas
          var smallCanvas = document.getElementById("smallCanvas");
          var smallCtx = smallCanvas.getContext("2d");

          // Draw the original canvas onto the 28x28 canvas (scaling down)
          smallCtx.drawImage(
            canvas,
            0,
            0,
            smallCanvas.width,
            smallCanvas.height,
          );

          // Get the image data from the 28x28 canvas
          var smallImageData = smallCtx.getImageData(
            0,
            0,
            smallCanvas.width,
            smallCanvas.height,
          );
          var smallData = smallImageData.data;
          console.log(smallData)

        }}
      />
    </>
  );
}
