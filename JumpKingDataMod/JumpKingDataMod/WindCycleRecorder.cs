using JumpKing;
using JumpKing.Player;
using Microsoft.Xna.Framework.Input;
using System;
using System.IO;
using System.Reflection;

namespace JumpKingDataMod
{
    public class WindCycleRecorder
    {
        private readonly string _outputPath;
        private bool _recording = false;
        private int _frameCount = 0;
        private const int MAX_FRAMES = 780; // 13 seconds at 60fps
        private KeyboardState _prevKeyState;

        public WindCycleRecorder()
        {
            _outputPath = Path.Combine(
                Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location),
                "wind_cycle.txt"
            );
            _prevKeyState = Keyboard.GetState();
        }

        public void Update(float windVelocity)
        {
            KeyboardState keyState = Keyboard.GetState();

            // press R to start recording
            if (keyState.IsKeyDown(Keys.R) && !_prevKeyState.IsKeyDown(Keys.R))
            {
                if (!_recording)
                {
                    _recording = true;
                    _frameCount = 0;
                    File.WriteAllText(_outputPath, "frame,wind_velocity\n");
                    Console.WriteLine("Wind cycle recording started");
                }
            }

            if (_recording)
            {
                File.AppendAllText(_outputPath, $"{_frameCount},{windVelocity:F6}\n");
                _frameCount++;

                if (_frameCount >= MAX_FRAMES)
                {
                    _recording = false;
                    Console.WriteLine("Wind cycle recording complete");
                }
            }

            _prevKeyState = keyState;
        }
    }
}