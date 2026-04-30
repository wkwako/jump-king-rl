using EntityComponent;
using JumpKing.Player;
using Microsoft.Xna.Framework;
using System;
using System.IO;
using System.Reflection;
using System.Threading;

namespace JumpKingDataMod
{
    public class TeleporterBehavior
    {
        private static string _teleportPath;
        private static FieldInfo _positionField;
        private static bool _initialized = false;
        private bool _isActive = true;

        public TeleporterBehavior()
        {
            if (!_initialized)
                Initialize();
        }

        private static void Initialize()
        {
            _teleportPath = Path.Combine(
                Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location),
                "teleport.txt"
            );

            // try to find position field on BodyComp via reflection
            _positionField = typeof(BodyComp).GetField("m_position",
                BindingFlags.NonPublic | BindingFlags.Instance);

            // fallback field names if m_position doesn't exist
            if (_positionField == null)
                _positionField = typeof(BodyComp).GetField("_position",
                    BindingFlags.NonPublic | BindingFlags.Instance);

            if (_positionField == null)
                _positionField = typeof(BodyComp).GetField("position",
                    BindingFlags.NonPublic | BindingFlags.Instance);

            _initialized = true;
        }

        public void Deactivate()
        {
            _isActive = false;
        }

        public void Update(BodyComp body)
        {
            if (!_isActive || body == null) return;
            if (!body.IsOnGround) return;

            try
            {
                if (!File.Exists(_teleportPath)) return;

                string content = File.ReadAllText(_teleportPath).Trim();
                if (string.IsNullOrEmpty(content) || content == "0") return;

                // reset to sentinel before applying so we don't teleport twice
                File.WriteAllText(_teleportPath, "0");

                string[] parts = content.Split(',');
                if (parts.Length < 2) return;

                float x = float.Parse(parts[0]);
                float y = float.Parse(parts[1]);

                ApplyPosition(body, new Vector2(x, y));
            }
            catch (Exception e)
            {
                try
                {
                    File.AppendAllText(_teleportPath + ".log", $"Teleport error: {e.Message}\n");
                }
                catch { }
            }
        }

        private static void ApplyPosition(BodyComp body, Vector2 newPos)
        {
            if (_positionField != null)
            {
                // set via reflection if we found the backing field
                _positionField.SetValue(body, newPos);

                // zero out velocity so physics state is clean
                FieldInfo velField = typeof(BodyComp).GetField("m_velocity",
                    BindingFlags.NonPublic | BindingFlags.Instance);
                if (velField == null)
                    velField = typeof(BodyComp).GetField("_velocity",
                        BindingFlags.NonPublic | BindingFlags.Instance);
                if (velField != null)
                    velField.SetValue(body, Vector2.Zero);
            }
        }
    }
}