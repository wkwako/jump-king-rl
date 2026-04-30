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
        private bool _wasOnGround = false;
        private float? _maxHeightThisJump = null;
        private bool _isActive = true;
        private bool _reflectionInitialized = false;

        private Type _jumpChargeCalcType;
        private PropertyInfo _jumpFramesProp;
        private PropertyInfo _jumpPercentageProp;
        private TeleporterBehavior _teleporter;

        private static readonly int[] WindScreens = { 25, 26, 27, 28, 29, 30, 31 };
        private static readonly int[] IceScreens = { 36, 37, 38 };

        public GameStateWriterBehaviour(PlayerEntity player)
        {
            _teleporter = new TeleporterBehavior();
        }

        private void InitializeReflection()
        {
            foreach (Assembly asm in AppDomain.CurrentDomain.GetAssemblies())
            {
                if (asm.GetName().Name == "JumpKingLastJumpValue")
                {
                    _jumpChargeCalcType = asm.GetType("JumpKingLastJumpValue.Models.JumpChargeCalc");
                    if (_jumpChargeCalcType != null)
                    {
                        _jumpFramesProp = _jumpChargeCalcType.GetProperty("JumpFrames", BindingFlags.Public | BindingFlags.Static);
                        _jumpPercentageProp = _jumpChargeCalcType.GetProperty("JumpPercentage", BindingFlags.Public | BindingFlags.Static);
                    }
                    break;
                }
            }
            _reflectionInitialized = true;
        }

        public void Deactivate()
        {
            _isActive = false;
            _teleporter?.Deactivate();
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

            if (!_reflectionInitialized)
                InitializeReflection();

            try
            {
                BodyComp body = context.BodyComp;
                if (body == null) return true;
                _teleporter?.Update(body);

                bool isOnGround = body.IsOnGround;
                int currentScreen = Camera.CurrentScreen;
                float x = body.Position.X;
                float y = body.Position.Y;
                float velX = body.Velocity.X;
                float velY = body.Velocity.Y;
                int totalScreens = LevelManager.TotalScreens;

                int jumpFrames = _jumpFramesProp != null
                    ? (int)_jumpFramesProp.GetValue(null)
                    : 0;
                float jumpPercentage = _jumpPercentageProp != null
                    ? (float)_jumpPercentageProp.GetValue(null)
                    : 0f;

                // reset on first airborne frame
                if (!isOnGround && _wasOnGround)
                {
                    _maxHeightThisJump = null;
                }

                // track max height (min Y) while airborne
                if (!isOnGround)
                {
                    if (_maxHeightThisJump == null || y < _maxHeightThisJump)
                        _maxHeightThisJump = y;
                }

                // fall back to current Y if no airborne frames recorded yet
                float maxHeight = _maxHeightThisJump ?? y;

                //get platform data
                PlatformScanner.ScanAndWrite(body.Position.X, body.Position.Y, currentScreen, totalScreens);

                // invert Y values so higher = larger number in Python
                string state = $"{x},{-y},{velX},{-velY},{isOnGround},{currentScreen},{totalScreens},{jumpFrames},{jumpPercentage},{-maxHeight}";

                WriteStateSafe(state);

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