using EntityComponent;
using EntityComponent.BT;
using JumpKing;
using JumpKing.API;
using JumpKing.BodyCompBehaviours;
using JumpKing.Level;
using JumpKing.Mods;
using JumpKing.Player;
using Microsoft.Xna.Framework;
using System;
using System.IO;
using System.Reflection;
using System.Threading;

namespace JumpKingDataMod
{
    [JumpKingMod("wkwako.GameStateMod")]
    public static class GameStateMod
    {
        public static string _outputPath;
        private static GameStateWriterBehaviour _behaviour;

        [OnLevelStart]
        public static void OnLevelStart()
        {
            _outputPath = Path.Combine(
                Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location),
                "gamestate.txt"
            );

            PlayerEntity player = EntityManager.instance.Find<PlayerEntity>();
            if (player == null) return;
            _behaviour = new GameStateWriterBehaviour(player);
            player.m_body.RegisterBehaviour(_behaviour);
        }

        [OnLevelEnd]
        public static void OnLevelEnd()
        {
            if (_behaviour != null)
                _behaviour.Deactivate();
        }
    }

    public class GameStateWriterBehaviour : IBodyCompBehaviour
    {
        private JumpState _jumpState;
        private FieldInfo _timerField;
        private bool _wasOnGround = false;
        private float? _maxHeightThisJump = null;
        private bool _isActive = true;

        private static readonly int[] WindScreens = { 25, 26, 27, 28, 29, 30, 31 };
        private static readonly int[] IceScreens = { 36, 37, 38 };

        public GameStateWriterBehaviour(PlayerEntity player)
        {
            FieldInfo jumpStateField = typeof(PlayerEntity).GetField("m_jump_state",
                BindingFlags.NonPublic | BindingFlags.Instance);
            _jumpState = (JumpState)jumpStateField.GetValue(player);

            _timerField = typeof(JumpState).GetField("m_timer",
                BindingFlags.NonPublic | BindingFlags.Instance);
        }

        public void Deactivate()
        {
            _isActive = false;
            try
            {
                if (File.Exists(GameStateMod._outputPath))
                    File.Delete(GameStateMod._outputPath);
            }
            catch
            {
                // ignore, shutting down
            }
        }

        private bool IsSpecialScreen(int screen)
        {
            return Array.IndexOf(WindScreens, screen) >= 0 ||
                   Array.IndexOf(IceScreens, screen) >= 0;
        }

        private void WriteStateSafe(string state)
        {
            int attempts = 0;
            while (attempts < 5)
            {
                try
                {
                    File.WriteAllText(GameStateMod._outputPath, state);
                    return;
                }
                catch (UnauthorizedAccessException)
                {
                    // access denied, stop trying
                    return;
                }
                catch (IOException)
                {
                    attempts++;
                    Thread.Sleep(5);
                }
            }
        }

        public bool ExecuteBehaviour(BehaviourContext context)
        {
            if (!_isActive) return true;

            try
            {
                BodyComp body = context.BodyComp;
                if (body == null) return true;

                bool isOnGround = body.IsOnGround;
                int currentScreen = Camera.CurrentScreen;
                float x = body.Position.X;
                float y = body.Position.Y;
                float velX = body.Velocity.X;
                float velY = body.Velocity.Y;
                int totalScreens = LevelManager.TotalScreens;
                float chargeTimer = _jumpState != null && _timerField != null
                    ? (float)_timerField.GetValue(_jumpState)
                    : 0f;

                // track max height (min Y) while airborne
                if (!isOnGround)
                {
                    if (_maxHeightThisJump == null || y < _maxHeightThisJump)
                        _maxHeightThisJump = y;
                }

                // fall back to current Y if no airborne frames recorded yet
                float maxHeight = _maxHeightThisJump ?? y;

                // invert Y values so higher = larger number in Python
                string state = $"{x},{-y},{velX},{-velY},{isOnGround},{currentScreen},{totalScreens},{chargeTimer},{-maxHeight}";

                if (IsSpecialScreen(currentScreen))
                {
                    // wind/ice: write every frame, Python handles race condition
                    WriteStateSafe(state);
                }
                else if (isOnGround && !_wasOnGround)
                {
                    // normal ground: write only on first frame of landing
                    WriteStateSafe(state);
                    _maxHeightThisJump = y; // reset to landing Y for next jump
                }

                _wasOnGround = isOnGround;
            }
            catch (Exception e)
            {
                WriteStateSafe($"ERROR:{e.Message}");
            }
            return true;
        }
    }
}