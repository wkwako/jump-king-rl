using EntityComponent;
using JumpKing;
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
        private bool _isActive = true;
        public TeleporterBehavior()
        {
            _teleportPath = Path.Combine(
                Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location),
                "teleport.txt"
            );
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

                File.WriteAllText(_teleportPath, "0");

                string[] parts = content.Split(',');
                if (parts.Length < 2) return;

                float x = float.Parse(parts[0]);
                float y = float.Parse(parts[1]);

                body.Position.X = x;
                body.Position.Y = y;
                body.Velocity = Vector2.Zero;
                Camera.UpdateCamera(body.GetHitbox().Center);
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
    }
}