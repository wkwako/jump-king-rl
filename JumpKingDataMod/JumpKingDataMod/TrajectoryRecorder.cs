using EntityComponent;
using JumpKing.Player;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Input;
using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;

namespace JumpKingDataMod
{
    public class TrajectoryRecorder
    {
        private readonly string _recordingPath;
        private KeyboardState _prevKeyState;

        private bool _isRecording = false;
        private bool _startCaptured = false;
        private float _startX;
        private float _startY;
        private int _spaceHeldFrames = 0;
        private List<string> _framePositions = new List<string>();

        private const float DELTA_TIME = 1f / 60f;

        public TrajectoryRecorder()
        {
            _recordingPath = Path.Combine(
                Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location),
                "trajectories.txt"
            );

            string timestamp = DateTime.Now.ToString("M/d/yyyy h:mmtt");
            File.AppendAllText(_recordingPath, $"Start session - {timestamp}\n");

            _prevKeyState = Keyboard.GetState();
        }

        public void Update(BodyComp body)
        {
            if (body == null) return;

            KeyboardState keyState = Keyboard.GetState();
            bool spaceDown = keyState.IsKeyDown(Keys.Space);
            bool prevSpaceDown = _prevKeyState.IsKeyDown(Keys.Space);
            bool isOnGround = body.IsOnGround;
            float x = body.Position.X;
            float y = body.Position.Y;

            // spacebar pressed — reset counters
            if (spaceDown && !prevSpaceDown && isOnGround)
            {
                _spaceHeldFrames = 0;
                _framePositions.Clear();
                _startCaptured = false;
                _isRecording = false;
            }

            // accumulate space held frames
            if (spaceDown)
                _spaceHeldFrames++;

            // spacebar released — start tracking airborne frames
            if (!spaceDown && prevSpaceDown)
                _isRecording = true;

            // capture start position on first airborne frame
            if (_isRecording && !isOnGround && !_startCaptured)
            {
                _startX = x;
                _startY = y;
                _startCaptured = true;
            }

            // record every airborne frame
            if (_isRecording && !isOnGround && _startCaptured)
                _framePositions.Add($"{x:F2},{-y:F2}");

            // landed — write full trajectory record
            if (_isRecording && isOnGround && _framePositions.Count > 0)
            {
                WriteRecord(x, y);
                _isRecording = false;
                _startCaptured = false;
                _framePositions.Clear();
            }

            _prevKeyState = keyState;
        }

        private void WriteRecord(float endX, float endY)
        {
            float spaceSeconds = _spaceHeldFrames * DELTA_TIME;
            string header = $"{_startX:F2},{-_startY:F2},{endX:F2},{-endY:F2},{_spaceHeldFrames},{spaceSeconds:F4}";
            string frames = string.Join("|", _framePositions);
            try
            {
                File.AppendAllText(_recordingPath, header + ";" + frames + "\n");
            }
            catch
            {
                // ignore write errors
            }
        }
    }
}