import { motion } from "framer-motion";

const leaves = [
  { x: "8%", y: "15%", size: 18, rotation: 30, delay: 0, duration: 6 },
  { x: "88%", y: "25%", size: 14, rotation: -45, delay: 1, duration: 7 },
  { x: "15%", y: "70%", size: 12, rotation: 60, delay: 2, duration: 5 },
  { x: "82%", y: "65%", size: 16, rotation: -20, delay: 0.5, duration: 8 },
  { x: "50%", y: "85%", size: 10, rotation: 45, delay: 1.5, duration: 6 },
  { x: "30%", y: "5%", size: 11, rotation: -60, delay: 3, duration: 7 },
  { x: "70%", y: "10%", size: 13, rotation: 15, delay: 2.5, duration: 5.5 },
];

export function FloatingLeaves() {
  return (
    <div className="pointer-events-none fixed inset-0 overflow-hidden z-0">
      {leaves.map((leaf, i) => (
        <motion.div
          key={i}
          className="absolute text-primary/[0.07]"
          style={{ left: leaf.x, top: leaf.y }}
          initial={{ opacity: 0 }}
          animate={{
            opacity: [0, 1, 1, 0],
            y: [0, -15, 5, 0],
            x: [0, 8, -5, 0],
            rotate: [leaf.rotation, leaf.rotation + 20, leaf.rotation - 10, leaf.rotation],
          }}
          transition={{
            duration: leaf.duration,
            delay: leaf.delay,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        >
          <svg
            width={leaf.size}
            height={leaf.size}
            viewBox="0 0 24 24"
            fill="currentColor"
            stroke="none"
          >
            <path d="M17 8C8 10 5.9 16.17 3.82 21.34l1.89.66.95-2.3c.48.17.98.3 1.34.3C19 20 22 3 22 3c-1 2-8 2.25-13 3.25S2 11.5 2 13.5s1.75 3.75 1.75 3.75C7 8 17 8 17 8z" />
          </svg>
        </motion.div>
      ))}
    </div>
  );
}
