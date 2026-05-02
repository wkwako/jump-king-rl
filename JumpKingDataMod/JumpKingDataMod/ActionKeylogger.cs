using System;
using System.IO;
using System.Reflection;
using Microsoft.Xna.Framework.Input;

namespace JumpKingDataMod
{
    public class ActionKeylogger
    {
        private readonly string _recordingPath;
        private float _leftHeldTime = 0f;
        private float _rightHeldTime = 0f;
        private float _spaceHeldTime = 0f;
        private bool _isRecording = false;
        private string _stateSnapshot = null;
        private KeyboardState _prevKeyState;

        private const float DELTA_TIME = 1f / 60f;

        public ActionKeylogger()
        {
            _recordingPath = Path.Combine(
                Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location),
                "recording.txt"
            );

            string timestamp = DateTime.Now.ToString("M/d/yyyy h:mmtt");
            File.AppendAllText(_recordingPath, $"Start session - {timestamp}\n");

            _prevKeyState = Keyboard.GetState();
        }

        public void Update(string currentState, bool isOnGround)
        {
            KeyboardState keyState = Keyboard.GetState();

            bool leftDown = keyState.IsKeyDown(Keys.Left);
            bool rightDown = keyState.IsKeyDown(Keys.Right);
            bool spaceDown = keyState.IsKeyDown(Keys.Space);
            bool anyDown = leftDown || rightDown || spaceDown;

            bool prevLeftDown = _prevKeyState.IsKeyDown(Keys.Left);
            bool prevRightDown = _prevKeyState.IsKeyDown(Keys.Right);
            bool prevSpaceDown = _prevKeyState.IsKeyDown(Keys.Space);
            bool prevAnyDown = prevLeftDown || prevRightDown || prevSpaceDown;

            // first key pressed — start new action, snapshot state if on ground
            if (anyDown && !prevAnyDown)
            {
                _isRecording = true;
                _leftHeldTime = 0f;
                _rightHeldTime = 0f;
                _spaceHeldTime = 0f;

                if (isOnGround)
                    _stateSnapshot = currentState;
            }

            // spacebar just pressed mid-action — re-snapshot if on ground
            if (spaceDown && !prevSpaceDown && _isRecording && isOnGround)
            {
                _stateSnapshot = currentState;
            }

            // accumulate durations
            if (_isRecording)
            {
                if (leftDown) _leftHeldTime += DELTA_TIME;
                if (rightDown) _rightHeldTime += DELTA_TIME;
                if (spaceDown) _spaceHeldTime += DELTA_TIME;
            }

            // all keys released — write record if we have a valid snapshot
            if (!anyDown && prevAnyDown && _isRecording && _stateSnapshot != null)
            {
                WriteRecord(_stateSnapshot, _leftHeldTime, _rightHeldTime, _spaceHeldTime);
                _isRecording = false;
                _stateSnapshot = null;
                _leftHeldTime = 0f;
                _rightHeldTime = 0f;
                _spaceHeldTime = 0f;
            }

            _prevKeyState = keyState;
        }

        private void WriteRecord(string stateSnapshot, float leftDuration, float rightDuration, float spaceDuration)
        {
            // collapse to single line for easy parsing
            string compactState = System.Text.RegularExpressions.Regex.Replace(stateSnapshot, @"\s+", "");
            string record = $"{compactState}|{leftDuration:F3},{rightDuration:F3},{spaceDuration:F3}\n";
            try
            {
                File.AppendAllText(_recordingPath, record);
            }
            catch
            {
                // ignore write errors
            }
        }
    }
}