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

namespace JumpKingDataMod
{
    [JumpKingMod("wkwako.GameStateMod")]
    public static class GameStateMod
    {

        public static string _outputPath;

        [OnLevelStart]
        public static void OnLevelStart()
        {
            _outputPath = Path.Combine(
            Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location),
            "gamestate.txt"
            );

            PlayerEntity player = EntityManager.instance.Find<PlayerEntity>();
            if (player == null) return;
            player.m_body.RegisterBehaviour(new GameStateWriterBehaviour(player));
        }

        [OnLevelEnd]
        public static void OnLevelEnd() { }
    }

    public class GameStateWriterBehaviour : IBodyCompBehaviour
    {
        private JumpState _jumpState;
        private FieldInfo _timerField;

        public GameStateWriterBehaviour(PlayerEntity player)
        {
            FieldInfo jumpStateField = typeof(PlayerEntity).GetField("m_jump_state",
                BindingFlags.NonPublic | BindingFlags.Instance);
            _jumpState = (JumpState)jumpStateField.GetValue(player);

            _timerField = typeof(JumpState).GetField("m_timer",
                BindingFlags.NonPublic | BindingFlags.Instance);
        }

        public bool ExecuteBehaviour(BehaviourContext context)
        {
            try
            {
                BodyComp body = context.BodyComp;
                if (body == null) return true;

                float x = body.Position.X;
                float y = body.Position.Y;
                float velX = body.Velocity.X;
                float velY = body.Velocity.Y;
                bool isOnGround = body.IsOnGround;
                int currentScreen = Camera.CurrentScreen;
                int totalScreens = LevelManager.TotalScreens;
                float chargeTimer = _jumpState != null && _timerField != null
                    ? (float)_timerField.GetValue(_jumpState)
                    : 0f;

                string state = $"{x},{y},{velX},{velY},{isOnGround},{currentScreen},{totalScreens},{chargeTimer}";
                File.WriteAllText(GameStateMod._outputPath, state);
            }
            catch (Exception e)
            {
                File.WriteAllText(GameStateMod._outputPath, $"ERROR:{e.Message}");
            }
            return true;
        }
    }
}